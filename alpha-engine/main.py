"""
Alpha Engine — FastAPI inference server.
Loads the trained XGBoost win-probability model and exposes it over HTTP
so the Rust execution engine can query it.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MODEL_PATH  = os.path.join(MODEL_DIR, "win_probability_model.json")
FEATS_PATH  = os.path.join(MODEL_DIR, "feature_columns.json")

app = FastAPI(title="NBA Alpha Engine", version="1.0.0")

# ── Load model at startup ────────────────────────────────────────────────────
model: xgb.XGBClassifier | None = None
feature_cols: list[str] = []


@app.on_event("startup")
def load_model():
    global model, feature_cols
    if not os.path.exists(MODEL_PATH):
        print(f"[alpha-engine] WARNING: model not found at {MODEL_PATH}. "
              "Run model.py first to train it.")
        return

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    with open(FEATS_PATH) as f:
        feature_cols = json.load(f)

    print(f"[alpha-engine] Loaded model with {len(feature_cols)} features.")


# ── Request / response schemas ───────────────────────────────────────────────
class GameState(BaseModel):
    """
    Live game state snapshot sent by the Rust engine.
    All numeric fields default to 0 so the caller only needs to
    supply what it actually knows — the model handles the rest.
    """
    # In-game state (most important at inference time)
    period: int             = 1
    game_seconds_left: float = 2880.0
    home_score: float       = 0.0
    away_score: float       = 0.0
    margin: float           = 0.0
    abs_margin: float       = 0.0
    total_points: float     = 0.0
    scoring_pace: float     = 0.0
    margin_x_time: float    = 0.0
    abs_margin_x_time: float = 0.0
    is_q4: int              = 0
    is_close_late: int      = 0
    is_blowout: int         = 0
    home_momentum_2min: float = 0.0
    away_momentum_2min: float = 0.0
    momentum_swing: float   = 0.0
    max_home_lead: float    = 0.0
    max_away_lead: float    = 0.0
    lead_volatility: float  = 0.0

    # Pre-game / season stats (optional — use 0 if unavailable live)
    home_pace: float        = 0.0
    away_pace: float        = 0.0
    home_net_rating: float  = 0.0
    away_net_rating: float  = 0.0
    diff_net_rating: float  = 0.0
    home_rest_days: float   = 2.0
    away_rest_days: float   = 2.0
    diff_rest_days: float   = 0.0
    home_is_b2b: int        = 0
    away_is_b2b: int        = 0


class PredictResponse(BaseModel):
    win_probability: float   # P(home team wins), in [0, 1]
    model_loaded: bool


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(state: GameState):
    if model is None:
        # No model yet → return a naive prior based on current margin
        naive_prob = float(np.clip(0.5 + state.margin * 0.01, 0.01, 0.99))
        return PredictResponse(win_probability=naive_prob, model_loaded=False)

    # Build feature vector aligned with training columns
    state_dict = state.model_dump()

    # Map Pydantic snake_case → UPPER_CASE training column names
    alias = {
        "period":             "PERIOD",
        "game_seconds_left":  "GAME_SECONDS_LEFT",
        "home_score":         "HOME_SCORE",
        "away_score":         "AWAY_SCORE",
        "margin":             "MARGIN",
        "abs_margin":         "ABS_MARGIN",
        "total_points":       "TOTAL_POINTS",
        "scoring_pace":       "SCORING_PACE",
        "margin_x_time":      "MARGIN_X_TIME",
        "abs_margin_x_time":  "ABS_MARGIN_X_TIME",
        "is_q4":              "IS_Q4",
        "is_close_late":      "IS_CLOSE_LATE",
        "is_blowout":         "IS_BLOWOUT",
        "home_momentum_2min": "HOME_MOMENTUM_2MIN",
        "away_momentum_2min": "AWAY_MOMENTUM_2MIN",
        "momentum_swing":     "MOMENTUM_SWING",
        "max_home_lead":      "MAX_HOME_LEAD",
        "max_away_lead":      "MAX_AWAY_LEAD",
        "lead_volatility":    "LEAD_VOLATILITY",
        "home_pace":          "HOME_PACE",
        "away_pace":          "AWAY_PACE",
        "home_net_rating":    "HOME_NET_RATING",
        "away_net_rating":    "AWAY_NET_RATING",
        "diff_net_rating":    "DIFF_NET_RATING",
        "home_rest_days":     "HOME_REST_DAYS",
        "away_rest_days":     "AWAY_REST_DAYS",
        "diff_rest_days":     "DIFF_REST_DAYS",
        "home_is_b2b":        "HOME_IS_B2B",
        "away_is_b2b":        "AWAY_IS_B2B",
    }
    mapped = {alias[k]: v for k, v in state_dict.items() if k in alias}

    # Fill missing features with 0
    row = np.array([mapped.get(col, 0.0) for col in feature_cols], dtype=np.float32)
    prob = float(model.predict_proba(row.reshape(1, -1))[0, 1])
    prob = float(np.clip(prob, 0.01, 0.99))

    return PredictResponse(win_probability=prob, model_loaded=True)
