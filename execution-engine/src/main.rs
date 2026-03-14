// ============================================================================
//  NBA Execution Engine — Module B
//  Polymarket live odds ingestion + shadow trading
//
//  Flow:
//    1. Query Polymarket REST API → discover live NBA markets
//    2. Open WebSocket to Polymarket CLOB → stream price ticks
//    3. On each tick → ask Alpha Engine for model probability
//    4. If |model_prob - market_prob| > EDGE_THRESHOLD → log shadow trade
//    5. All trades land in SQLite `simulated_trades` table
// ============================================================================

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{error, info, warn};
use uuid::Uuid;

// ── Configuration ────────────────────────────────────────────────────────────

const POLYMARKET_REST:   &str = "https://clob.polymarket.com";
const POLYMARKET_WS:     &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const ALPHA_ENGINE_URL:  &str = "http://127.0.0.1:8000";
const DB_PATH:           &str = "../trades.sqlite";

/// Minimum edge (model_prob - market_prob) required to trigger a shadow trade.
const EDGE_THRESHOLD: f64 = 0.05;

/// Simulated USDC stake per trade.
const STAKE_USDC: f64 = 50.0;

// ── Polymarket REST types ─────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct MarketsResponse {
    data:        Vec<Market>,
    next_cursor: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct Market {
    condition_id: String,
    question:     String,
    #[allow(dead_code)]
    active:       bool,
    closed:       bool,
    tokens:       Vec<Token>,
}

#[derive(Debug, Deserialize, Clone)]
struct Token {
    token_id: String,
    outcome:  String, // "Yes" or "No"
    /// Snapshot price from REST — used as a fallback before WebSocket ticks arrive.
    #[allow(dead_code)]
    price:    f64,
}

// ── WebSocket message types ───────────────────────────────────────────────────

/// Outbound: subscribe to price-change events for a list of asset IDs.
#[derive(Serialize)]
struct WsSubscribe {
    #[serde(rename = "type")]
    msg_type:  &'static str,
    channel:   &'static str,
    assets_ids: Vec<String>,
}

// Polymarket WS messages are parsed directly as serde_json::Value
// because the schema varies between price_change, book, and last_trade_price events.

// ── Alpha Engine types ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct GameStateRequest {
    period:           i32,
    game_seconds_left: f64,
    home_score:       f64,
    away_score:       f64,
    margin:           f64,
    abs_margin:       f64,
    total_points:     f64,
    scoring_pace:     f64,
    // remaining fields default to 0 — FastAPI fills them
}

#[derive(Deserialize)]
struct PredictResponse {
    win_probability: f64,
    model_loaded:    bool,
}

// ── In-memory market registry ─────────────────────────────────────────────────

/// Maps a Polymarket token_id → enough context to log a meaningful trade.
#[derive(Debug, Clone)]
struct MarketEntry {
    condition_id: String,
    question:     String,
    #[allow(dead_code)]
    token_id:     String,
    outcome:      String, // "Yes" or "No"
}

// ── SQLite helpers ────────────────────────────────────────────────────────────

fn init_db(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS simulated_trades (
            id                  TEXT PRIMARY KEY,
            timestamp           TEXT NOT NULL,
            game_id             TEXT NOT NULL,
            target_team         TEXT NOT NULL,
            action              TEXT NOT NULL,
            market_implied_prob REAL NOT NULL,
            model_implied_prob  REAL NOT NULL,
            stake_amount        REAL NOT NULL,
            status              TEXT NOT NULL DEFAULT 'OPEN',
            pnl                 REAL
        );",
    )?;
    info!("SQLite ready at {DB_PATH}");
    Ok(())
}

fn log_trade(
    conn:        &Connection,
    game_id:     &str,
    target_team: &str,
    action:      &str,      // "BUY_YES" | "BUY_NO"
    market_prob: f64,
    model_prob:  f64,
) -> Result<()> {
    let id        = Uuid::new_v4().to_string();
    let timestamp = Utc::now().to_rfc3339();

    conn.execute(
        "INSERT INTO simulated_trades
            (id, timestamp, game_id, target_team, action,
             market_implied_prob, model_implied_prob, stake_amount, status, pnl)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'OPEN', NULL)",
        params![
            id, timestamp, game_id, target_team, action,
            market_prob, model_prob, STAKE_USDC
        ],
    )?;

    info!(
        "TRADE LOGGED  id={id}  game={game_id}  action={action}  \
         market={market_prob:.3}  model={model_prob:.3}  \
         edge={:.3}  stake={STAKE_USDC} USDC",
        model_prob - market_prob
    );
    Ok(())
}

// ── Polymarket: fetch live NBA markets ───────────────────────────────────────

async fn fetch_nba_markets(http: &Client) -> Result<Vec<Market>> {
    let mut nba_markets: Vec<Market> = Vec::new();
    let mut cursor = String::new();

    loop {
        let url = if cursor.is_empty() {
            format!("{POLYMARKET_REST}/markets?limit=100&active=true")
        } else {
            format!("{POLYMARKET_REST}/markets?limit=100&active=true&next_cursor={cursor}")
        };

        let resp: MarketsResponse = http
            .get(&url)
            .send()
            .await
            .context("GET /markets failed")?
            .json()
            .await
            .context("deserialise /markets failed")?;

        let page_count = resp.data.len();

        for m in resp.data {
            let q = m.question.to_lowercase();
            // Filter: active NBA game-winner markets
            if !m.closed
                && (q.contains("nba") || q.contains("will the") && q.contains("win"))
                && (q.contains("win") || q.contains("champion") || q.contains("beat"))
            {
                nba_markets.push(m);
            }
        }

        match resp.next_cursor {
            Some(c) if !c.is_empty() && page_count == 100 => cursor = c,
            _ => break,
        }
    }

    info!("Found {} NBA markets on Polymarket", nba_markets.len());
    Ok(nba_markets)
}

// ── Alpha Engine: get win probability ────────────────────────────────────────

async fn get_model_prob(http: &Client, game_state: &GameStateRequest) -> Result<f64> {
    let resp: PredictResponse = http
        .post(format!("{ALPHA_ENGINE_URL}/predict"))
        .json(game_state)
        .send()
        .await
        .context("POST /predict failed")?
        .json()
        .await
        .context("deserialise /predict failed")?;

    if !resp.model_loaded {
        warn!("Alpha Engine returned naive prior (model not trained yet)");
    }
    Ok(resp.win_probability)
}

// ── Parse price from a JSON value (string or number) ─────────────────────────

fn parse_price(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

// ── WebSocket ingestion loop ──────────────────────────────────────────────────

async fn run_ws_ingestion(
    markets:  Vec<Market>,
    http:     Client,
    db:       Arc<Mutex<Connection>>,
) -> Result<()> {
    // Build token_id → MarketEntry lookup
    let mut registry: std::collections::HashMap<String, MarketEntry> =
        std::collections::HashMap::new();

    let mut all_token_ids: Vec<String> = Vec::new();

    for m in &markets {
        for t in &m.tokens {
            registry.insert(
                t.token_id.clone(),
                MarketEntry {
                    condition_id: m.condition_id.clone(),
                    question:     m.question.clone(),
                    token_id:     t.token_id.clone(),
                    outcome:      t.outcome.clone(),
                },
            );
            all_token_ids.push(t.token_id.clone());
        }
    }

    if all_token_ids.is_empty() {
        warn!("No NBA markets found — WebSocket will subscribe to an empty list. \
               Check market filters or try again later.");
        return Ok(());
    }

    info!(
        "Connecting to Polymarket WebSocket — subscribing to {} tokens across {} markets",
        all_token_ids.len(),
        markets.len()
    );

    let (mut ws_stream, _) = connect_async(POLYMARKET_WS)
        .await
        .context("WebSocket connect failed")?;

    info!("WebSocket connected");

    // Send subscription
    let sub = WsSubscribe {
        msg_type:   "subscribe",
        channel:    "price_change",
        assets_ids: all_token_ids,
    };
    let sub_json = serde_json::to_string(&sub)?;
    ws_stream.send(Message::Text(sub_json)).await?;
    info!("Subscription sent");

    // ── Main event loop ──
    while let Some(msg) = ws_stream.next().await {
        let msg = match msg {
            Ok(m)  => m,
            Err(e) => { error!("WS recv error: {e}"); break; }
        };

        let text = match msg {
            Message::Text(t)  => t,
            Message::Ping(d)  => {
                // respond to keepalive pings
                if let Err(e) = ws_stream.send(Message::Pong(d)).await {
                    error!("pong failed: {e}");
                }
                continue;
            }
            Message::Close(_) => {
                warn!("WebSocket closed by server");
                break;
            }
            _ => continue,
        };

        // Parse the raw JSON
        let raw: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v)  => v,
            Err(e) => { warn!("JSON parse error: {e} — raw: {text}"); continue; }
        };

        // Normalise into a flat list of ticks
        let ticks: Vec<serde_json::Value> = if raw.is_array() {
            raw.as_array().cloned().unwrap_or_default()
        } else {
            vec![raw]
        };

        for tick_val in ticks {
            // We only care about price_change events
            let event_type = tick_val
                .get("event_type")
                .or_else(|| tick_val.get("type"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if !event_type.contains("price") && !event_type.is_empty() {
                continue; // skip order-book / book snapshots
            }

            let asset_id = match tick_val.get("asset_id").and_then(|v| v.as_str()) {
                Some(id) => id.to_string(),
                None     => continue,
            };

            let market_prob = match tick_val.get("price").and_then(parse_price) {
                Some(p) => p,
                None    => continue,
            };

            // Clamp to sane probability range
            if !(0.01..=0.99).contains(&market_prob) {
                continue;
            }

            let entry = match registry.get(&asset_id) {
                Some(e) => e.clone(),
                None    => continue,
            };

            info!(
                "Tick  market=\"{}\"  outcome={}  market_prob={:.3}",
                entry.question, entry.outcome, market_prob
            );

            // We model P(home wins). Map YES token → home win probability.
            // Polymarket questions are typically "Will [HOME TEAM] win?"
            // so YES.price ≈ P(home wins).
            if entry.outcome.to_lowercase() != "yes" {
                continue; // use the YES side as our reference
            }

            // Build a minimal game-state for the alpha engine.
            // With no live PBP data yet we use a pre-game snapshot (all zeros).
            let game_state = GameStateRequest {
                period:            1,
                game_seconds_left: 2880.0,
                home_score:        0.0,
                away_score:        0.0,
                margin:            0.0,
                abs_margin:        0.0,
                total_points:      0.0,
                scoring_pace:      0.0,
            };

            let model_prob = match get_model_prob(&http, &game_state).await {
                Ok(p)  => p,
                Err(e) => {
                    warn!("Alpha Engine error: {e}");
                    // Fallback: skip trade rather than log garbage
                    continue;
                }
            };

            let edge = model_prob - market_prob;

            if edge.abs() < EDGE_THRESHOLD {
                continue; // No mispricing worth trading
            }

            // Determine action
            let action = if edge > 0.0 { "BUY_YES" } else { "BUY_NO" };

            info!(
                "EDGE FOUND  question=\"{}\"  edge={edge:.3}  action={action}",
                entry.question
            );

            // Log shadow trade
            let db_guard = db.lock().await;
            if let Err(e) = log_trade(
                &db_guard,
                &entry.condition_id,
                &entry.question,
                action,
                market_prob,
                model_prob,
            ) {
                error!("DB write failed: {e}");
            }
        }
    }

    Ok(())
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("execution_engine=info".parse()?)
                .add_directive("info".parse()?),
        )
        .init();

    info!("NBA Execution Engine starting...");

    // ── Database
    let conn = Connection::open(DB_PATH).context("open SQLite")?;
    init_db(&conn)?;
    let db = Arc::new(Mutex::new(conn));

    // ── HTTP client (shared for all calls)
    let http = Client::builder()
        .user_agent("nba-execution-engine/0.1")
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    // ── Check Alpha Engine health
    match http
        .get(format!("{ALPHA_ENGINE_URL}/health"))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => {
            let body: serde_json::Value = r.json().await.unwrap_or_default();
            info!("Alpha Engine healthy: {body}");
        }
        Ok(r) => warn!("Alpha Engine responded with HTTP {}", r.status()),
        Err(e) => warn!("Alpha Engine unreachable ({e}). Trades will use naive prior."),
    }

    // ── Discover NBA markets
    let markets = fetch_nba_markets(&http).await?;

    if markets.is_empty() {
        warn!("No active NBA markets on Polymarket right now. \
               The engine will retry in 60 s…");
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        // In production: loop and retry. For MVP we exit and let the operator restart.
        return Ok(());
    }

    // ── Launch WebSocket ingestion loop
    // Reconnects automatically on drop — in production wrap in a retry loop.
    loop {
        info!("(Re)connecting WebSocket ingestion loop…");
        match run_ws_ingestion(markets.clone(), http.clone(), Arc::clone(&db)).await {
            Ok(_)  => warn!("WS loop exited cleanly — reconnecting in 5 s"),
            Err(e) => error!("WS loop error: {e} — reconnecting in 5 s"),
        }
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    }
}
