# NBA Shadow Trader

Three services, one command each. Run them in order.

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.11 |
| Rust + Cargo | ≥ 1.78 |
| Node.js | ≥ 20 |

---

## 1 — Alpha Engine (Python · port 8000)

```bash
cd alpha-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train the model first (one-time, ~5 min)
python ../../model.py

# Start inference server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Health check: http://localhost:8000/health

---

## 2 — Execution Engine (Rust · writes to ../trades.sqlite)

```bash
cd execution-engine
cargo run --release
```

Reads from Polymarket WebSocket, queries Alpha Engine, writes shadow trades to `trades.sqlite`.

Env vars (optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_THRESHOLD` | `0.05` | Minimum model–market spread to trigger a trade |
| `RUST_LOG` | `info` | Log level (`debug` for verbose) |

```bash
RUST_LOG=debug cargo run --release
```

---

## 3 — Dashboard (Next.js · port 3000)

```bash
cd dashboard
npm install
npm run dev
```

Open http://localhost:3000 — polls the backend every 3 s.

---

## Data flow

```
Polymarket WS ──► Execution Engine ──► SQLite (trades.sqlite)
                        │                       ▲
                        ▼                       │
                  Alpha Engine (Python)   Dashboard (Next.js)
                  POST /predict           GET /api/trades
```

---

## Project structure

```
nba-trading-bot/
├── alpha-engine/        # FastAPI win-probability server
│   ├── main.py
│   └── requirements.txt
├── execution-engine/    # Rust WebSocket + shadow trader
│   ├── Cargo.toml
│   └── src/main.rs
├── dashboard/           # Next.js real-time UI
│   ├── package.json
│   ├── next.config.js
│   └── src/app/
├── .gitignore
└── README.md
```
