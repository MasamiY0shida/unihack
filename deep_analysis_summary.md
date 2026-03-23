# Courtside Alpha — Deep Analysis Summary

## Data Scope
- **511 total trades**, 215 resolved | **12,081 live snapshots** across 59 games
- **Total PnL: +$277.17** | Win rate: 51.6% | Avg PnL/trade: $1.29
- Best trade: +$221.67 | Worst: -$46.88

---

## 1. Calibration — Model vs Reality

| Model predicts | Actual win rate | Verdict |
|---|---|---|
| ~6% | 50% | Massively underconfident |
| ~17% | 18% | Correct |
| ~25% | 48% | Underconfident |
| ~36% | 73% | Very underconfident |
| ~46% | 37% | Overconfident |
| ~55% | 59% | Correct |
| ~64% | 64% | **Correct** |
| ~74% | 48% | Very overconfident |
| ~84% | 63% | Overconfident |
| ~93% | 52% | Catastrophically overconfident |

**Takeaway:** Model is well-calibrated only in the 55-65% range. At both extremes (<20% and >80%), predictions are unreliable. When model says 93%, actual outcome is a coin flip.

---

## 2. Feature Importances (XGBoost gain)

**Top 10 features driving win probability:**

| Rank | Feature | Category | Gain |
|---|---|---|---|
| 1 | MARGIN | Game state | 568 |
| 2 | MARGIN_X_PROGRESS | Game state | 306 |
| 3 | POSSESSIONS_TO_CLOSE | v4 new | 168 |
| 4 | MARGIN_X_TIME_DECAY | v4 new | 117 |
| 5 | ABS_MARGIN | Game state | 115 |
| 6 | SWING_60s | Game state | 109 |
| 7 | MARGIN_OVER_SQRT_TIME | v4 fixed | 107 |
| 8 | ABS_MARGIN_X_PROGRESS | Game state | 93 |
| 9 | IS_CLOSE_LATE | Game state | 93 |
| 10 | SWING_300s | Game state | 81 |

**Category breakdown:**
- Game state: 32 features, 2019 total gain (dominant)
- Static pregame: 157 features, 1713 total gain (high count, low per-feature)
- Live boxscore: 48 features, 1516 total gain (noisy contributors)
- v4 new features: 9 features, 597 total gain (high per-feature value)

**Takeaway:** v4 features (POSSESSIONS_TO_CLOSE, MARGIN_X_TIME_DECAY) are #3 and #4 — working as intended. Live boxscore features like LIVE_STAR_FOUL_DANGER and LIVE_DIFF_FOULS are in top 15 and may cause early-game noise.

---

## 3. Game Time Series (3 case studies)

| Game | Trades | Resolved | PnL |
|---|---|---|---|
| SAC vs UTA (Kings) | 46 | 18 | **+$200.26** |
| HOU vs LAL (Rockets) | 34 | 17 | **-$59.57** |
| LAC vs SAC (Clippers) | 28 | 11 | **+$36.30** |

**Takeaway:** Model oscillates 3x more than market. In the Kings game it found a genuine edge (Kings underpriced). In Rockets game, the model whipsawed between sides causing repeated losses.

---

## 4. Trade Sequencing

- **45 unique games traded**, avg 11.4 trades/game
- **204 round-trips** (buy → sell → buy again) across 32 games
- **Correlation: trade count vs PnL = +0.53** (more trades = more profit, surprisingly)
- Games WITH round-trips: avg **+$9.76** PnL
- Games WITHOUT round-trips: avg **-$2.69** PnL

**Top games by volume:**

| Game | Trades | B/S | Round-trips | PnL |
|---|---|---|---|---|
| Trail Blazers vs 76ers | 62 | 35B/27S | 26 | -$6.57 |
| Jazz vs Kings | 46 | 27B/19S | 18 | +$200.26 |
| Bucks vs Suns | 38 | 19B/19S | 18 | +$219.55 |
| Warriors vs Knicks | 37 | 24B/13S | 12 | +$63.84 |

**Takeaway:** Round-tripping is NOT destroying value — the bot profits from frequent trading when edges are real. The problem is entering games where there's no genuine edge.

---

## 5. When Do Winning Trades Happen?

| Quarter | Count | % of total | Win Rate | PnL |
|---|---|---|---|---|
| Q1 | 11 | 6% | 45.5% | -$33 |
| Q2 | 18 | 10% | 44.4% | +$152 |
| **Q3** | **59** | **32%** | **61.0%** | **+$152** |
| Q4 | 94 | 52% | 47.9% | -$15 |

**Margin at entry:**
- Winners enter with avg **+2.1** margin (betting WITH the scoreboard)
- Losers enter with avg **-3.0** margin (betting AGAINST the scoreboard)

**Takeaway:** Q3 is the sweet spot — 61% win rate, +$152. Q4 has the most trades but worst win rate. The model should be more selective in Q4, or only enter when margin is positive.

---

## 6. Liquidity & Execution

- Bid-ask spread data: not available (bid == ask in snapshots)
- Avg Polymarket volume per game: **$2.3M** (good liquidity)
- Mean |model - market| gap: **22.3%** (large claimed edges)

**Takeaway:** Liquidity is fine. The large model-market gap is a calibration issue, not a real edge — if edges were truly 22% on average, win rate would be much higher.

---

## 7. Sub-Model Volatility

| Sub-Model | Avg tick swing | Max swing | Role |
|---|---|---|---|
| Win Prob | 1.78% | 93.8% | Main prediction |
| Proxy Prob | 0.009% | 7.2% | Market baseline |
| Margin | 0.74 pts | 54.4 pts | Score prediction |
| Edge | 1.79% | 93.8% | Win - Proxy |
| Edge Confidence | 1.45% | 48.6% | Signal gating |

**Correlation insight:** Edge confidence has near-zero correlation with everything else (0.094 with edge). It's not providing useful gating — the bot trades regardless of confidence level.

**Takeaway:** Win Prob and Edge swing wildly (max 94% in one tick), while Proxy is nearly static. The live model is the sole source of volatility. Edge confidence is decorrelated from actual edge — it should either be recalibrated or replaced.

---

## Key Recommendations

1. **Cap extreme predictions** — clip model output to 10-90% range until calibration improves
2. **Weight Q3 trades more heavily** — 61% win rate vs 45-48% in other quarters
3. **Only enter when margin is positive** — winners bet with the scoreboard (+2.1), losers bet against (-3.0)
4. **Fix or remove edge_confidence** — currently uncorrelated with actual outcomes
5. **Reduce live boxscore feature weight in Q1/Q2** — these features are noisy on small samples
