"""
Bias-Variance Decomposition for Courtside Alpha v4 Model
=========================================================
Uses bootstrap resampling (game-level) to decompose prediction error into:
  - Bias²:      (mean_prediction - true_outcome)² — systematic error
  - Variance:   var(predictions across bootstraps)  — instability
  - Noise:      irreducible (Bayes error)

Outputs a clean summary to stdout + saves plots.
"""

import sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

# ── Re-use model_v4's data pipeline ──────────────────────────────────────
# We import the heavy lifting from model_v4 to avoid duplicating 1500 lines
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("  BIAS-VARIANCE DECOMPOSITION — Courtside Alpha v4")
print("=" * 70)

# ── Step 1: Load and prepare data exactly as model_v4 does ───────────────
print("\n[1/5] Loading training data (replicating v4 pipeline)...")

from model_v4 import (
    load_all_data,
    build_rolling_team_features,
    build_team_profiles,
    extract_game_snapshots,
    enrich_snapshots_with_boxscore,
    damp_live_features,
    drop_mismatched_features,
    merge_all_features,
    compute_sample_weights,
    get_pregame_features,
    get_live_features,
    merge_live_observations,
)

data = load_all_data()
games_rolling = build_rolling_team_features(data["games"])
team_profiles = build_team_profiles(data)
snapshots = extract_game_snapshots(data["pbp"], games_rolling, snapshot_interval=90)
snapshots = enrich_snapshots_with_boxscore(snapshots, data["pbp"], data["games"])
snapshots = damp_live_features(snapshots)
snapshots = drop_mismatched_features(snapshots)
df = merge_all_features(snapshots, games_rolling, team_profiles, data["fatigue"])
df = compute_sample_weights(df, data["games"])
import os
if os.path.exists("live_observations.sqlite"):
    df = merge_live_observations(df)

pregame_feats = get_pregame_features(df)
live_feats = get_live_features(df)

print(f"  Data: {len(df)} rows, {df['GAME_ID'].nunique()} games")
print(f"  Pregame features: {len(pregame_feats)}")
print(f"  Live features: {len(live_feats)}")

# ── Step 2: Bootstrap parameters ─────────────────────────────────────────
N_BOOTSTRAPS = 20
BOOTSTRAP_FRAC = 0.8   # use 80% of games per bootstrap
RANDOM_SEED_BASE = 100

unique_games = df["GAME_ID"].unique()
n_games = len(unique_games)
n_sample = int(n_games * BOOTSTRAP_FRAC)

print(f"\n[2/5] Running {N_BOOTSTRAPS} bootstrap iterations...")
print(f"  {n_games} total games, sampling {n_sample} per bootstrap (game-level)")

# Storage for predictions on ALL rows across bootstraps
all_preds_win = np.full((N_BOOTSTRAPS, len(df)), np.nan)
all_preds_margin = np.full((N_BOOTSTRAPS, len(df)), np.nan)
all_preds_proxy = np.full((N_BOOTSTRAPS, len(df)), np.nan)

y_true_win = df["HOME_WON"].values.astype(float)
y_true_margin = df["FINAL_MARGIN"].values.astype(float)

for b in range(N_BOOTSTRAPS):
    rng = np.random.RandomState(RANDOM_SEED_BASE + b)

    # Sample games WITH replacement (bootstrap)
    boot_games = rng.choice(unique_games, size=n_sample, replace=True)
    boot_game_set = set(boot_games)
    oob_game_set = set(unique_games) - boot_game_set

    train_mask = df["GAME_ID"].isin(boot_game_set)
    oob_mask = df["GAME_ID"].isin(oob_game_set)

    if oob_mask.sum() == 0:
        continue

    X_train_pre = df.loc[train_mask, pregame_feats].values
    X_train_live = df.loc[train_mask, live_feats].values
    y_train = y_true_win[train_mask]
    y_train_margin = y_true_margin[train_mask]
    w_train = df.loc[train_mask, "SAMPLE_WEIGHT"].values

    X_oob_pre = df.loc[oob_mask, pregame_feats].values
    X_oob_live = df.loc[oob_mask, live_feats].values

    # Train proxy model (pregame features only)
    proxy = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=20, reg_alpha=0.5, reg_lambda=2.0,
        random_state=RANDOM_SEED_BASE + b, verbosity=0,
    )
    proxy.fit(X_train_pre, y_train, sample_weight=w_train)

    # Train live win probability model
    live_cls = xgb.XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.5,
        min_child_weight=10, reg_alpha=0.2, reg_lambda=1.5, gamma=0.1,
        random_state=RANDOM_SEED_BASE + b, verbosity=0,
    )
    live_cls.fit(X_train_live, y_train, sample_weight=w_train)

    # Train margin regressor
    live_reg = xgb.XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.5,
        min_child_weight=10, reg_alpha=0.2, reg_lambda=1.5, gamma=0.1,
        random_state=RANDOM_SEED_BASE + b, verbosity=0,
    )
    live_reg.fit(X_train_live, y_train_margin, sample_weight=w_train)

    # Predict on OOB samples only
    oob_idx = np.where(oob_mask.values)[0]
    all_preds_proxy[b, oob_idx] = proxy.predict_proba(X_oob_pre)[:, 1]
    all_preds_win[b, oob_idx] = live_cls.predict_proba(X_oob_live)[:, 1]
    all_preds_margin[b, oob_idx] = live_reg.predict(X_oob_live)

    n_oob_games = len(oob_game_set)
    print(f"  Bootstrap {b+1}/{N_BOOTSTRAPS}: "
          f"train={train_mask.sum()} rows ({len(boot_game_set)} games), "
          f"OOB={oob_mask.sum()} rows ({n_oob_games} games)")

# ── Step 3: Compute bias-variance decomposition ─────────────────────────
print(f"\n[3/5] Computing bias-variance decomposition...")

def bias_variance_decomp(preds_matrix, y_true, model_name, is_classifier=True):
    """
    For each sample, compute:
      - mean prediction across bootstraps (where it appeared in OOB)
      - variance of predictions
      - bias² = (mean_pred - y_true)²
    Only use samples that appeared in at least 3 OOB sets.
    """
    n_samples = preds_matrix.shape[1]

    mean_preds = np.full(n_samples, np.nan)
    var_preds = np.full(n_samples, np.nan)
    bias_sq = np.full(n_samples, np.nan)
    counts = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        valid = ~np.isnan(preds_matrix[:, i])
        counts[i] = valid.sum()
        if counts[i] >= 3:
            preds_i = preds_matrix[valid, i]
            mean_preds[i] = preds_i.mean()
            var_preds[i] = preds_i.var()
            bias_sq[i] = (mean_preds[i] - y_true[i]) ** 2

    valid_mask = counts >= 3
    n_valid = valid_mask.sum()

    avg_bias_sq = np.nanmean(bias_sq[valid_mask])
    avg_variance = np.nanmean(var_preds[valid_mask])

    if is_classifier:
        # For binary classification, irreducible noise = p(1-p)
        avg_noise = np.nanmean(mean_preds[valid_mask] * (1 - mean_preds[valid_mask]))
        total_error = avg_bias_sq + avg_variance + avg_noise
    else:
        # For regression, compute MSE directly
        mse = np.nanmean((mean_preds[valid_mask] - y_true[valid_mask]) ** 2)
        avg_noise = 0  # can't separate for regression without oracle
        total_error = mse

    print(f"\n  {model_name}")
    print(f"  {'─' * 50}")
    print(f"  Valid samples:   {n_valid} / {n_samples}")
    print(f"  Avg OOB count:   {counts[valid_mask].mean():.1f} bootstraps per sample")
    print(f"  Bias²:           {avg_bias_sq:.6f}")
    print(f"  Variance:        {avg_variance:.6f}")
    if is_classifier:
        print(f"  Noise (p(1-p)):  {avg_noise:.6f}")
    print(f"  Total Error:     {total_error:.6f}")
    print(f"  Bias²/Var ratio: {avg_bias_sq / max(avg_variance, 1e-10):.2f}")

    if avg_bias_sq > avg_variance * 2:
        diagnosis = "HIGH BIAS (underfitting) — model systematically wrong"
    elif avg_variance > avg_bias_sq * 2:
        diagnosis = "HIGH VARIANCE (overfitting) — model unstable across samples"
    else:
        diagnosis = "BALANCED — neither dominates"
    print(f"  Diagnosis:       {diagnosis}")

    return {
        "model": model_name,
        "n_valid": n_valid,
        "bias_sq": avg_bias_sq,
        "variance": avg_variance,
        "noise": avg_noise,
        "total": total_error,
        "mean_preds": mean_preds,
        "var_preds": var_preds,
        "bias_sq_per_sample": bias_sq,
        "valid_mask": valid_mask,
        "counts": counts,
    }

results = {}
results["win_prob"] = bias_variance_decomp(all_preds_win, y_true_win, "Win Probability (classifier)", is_classifier=True)
results["margin"] = bias_variance_decomp(all_preds_margin, y_true_margin, "Margin Regressor", is_classifier=False)
results["proxy"] = bias_variance_decomp(all_preds_proxy, y_true_win, "Market Proxy (pregame)", is_classifier=True)

# ── Step 4: Breakdown by game stage ──────────────────────────────────────
print(f"\n[4/5] Bias-variance by game stage...")
print(f"\n  {'Stage':<20} {'Bias²':>10} {'Variance':>10} {'Ratio B/V':>10} {'Diagnosis'}")
print(f"  {'─'*70}")

game_progress = df["GAME_PROGRESS"].values if "GAME_PROGRESS" in df.columns else None
if game_progress is not None:
    stages = [
        ("Q1 (0-25%)", 0.0, 0.25),
        ("Q2 (25-50%)", 0.25, 0.50),
        ("Q3 (50-75%)", 0.50, 0.75),
        ("Q4 early (75-90%)", 0.75, 0.90),
        ("Q4 late (90-100%)", 0.90, 1.01),
    ]

    stage_results = []
    for label, lo, hi in stages:
        stage_mask = (game_progress >= lo) & (game_progress < hi)
        valid = results["win_prob"]["valid_mask"] & stage_mask
        if valid.sum() < 10:
            continue

        b2 = np.nanmean(results["win_prob"]["bias_sq_per_sample"][valid])
        v = np.nanmean(results["win_prob"]["var_preds"][valid])
        ratio = b2 / max(v, 1e-10)

        if b2 > v * 2:
            diag = "HIGH BIAS"
        elif v > b2 * 2:
            diag = "HIGH VARIANCE"
        else:
            diag = "balanced"

        print(f"  {label:<20} {b2:>10.6f} {v:>10.6f} {ratio:>10.2f} {diag}")
        stage_results.append({"stage": label, "bias_sq": b2, "variance": v})

# ── Step 5: Breakdown by prediction confidence ───────────────────────────
print(f"\n  Bias-variance by prediction confidence (win_prob model):")
print(f"  {'Pred Range':<20} {'Bias²':>10} {'Variance':>10} {'Ratio B/V':>10} {'N':>6}")
print(f"  {'─'*60}")

mean_p = results["win_prob"]["mean_preds"]
valid = results["win_prob"]["valid_mask"]

conf_bins = [
    ("Extreme low <15%", 0.0, 0.15),
    ("Low 15-30%", 0.15, 0.30),
    ("Mid-low 30-45%", 0.30, 0.45),
    ("Toss-up 45-55%", 0.45, 0.55),
    ("Mid-high 55-70%", 0.55, 0.70),
    ("High 70-85%", 0.70, 0.85),
    ("Extreme high >85%", 0.85, 1.01),
]

for label, lo, hi in conf_bins:
    mask = valid & (~np.isnan(mean_p)) & (mean_p >= lo) & (mean_p < hi)
    n = mask.sum()
    if n < 5:
        continue
    b2 = np.nanmean(results["win_prob"]["bias_sq_per_sample"][mask])
    v = np.nanmean(results["win_prob"]["var_preds"][mask])
    ratio = b2 / max(v, 1e-10)
    print(f"  {label:<20} {b2:>10.6f} {v:>10.6f} {ratio:>10.2f} {n:>6}")

# ── Step 6: Generate plots ───────────────────────────────────────────────
print(f"\n[5/5] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#1a1a2e")
for ax in axes.flat:
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#334155")

# Plot 1: Bias² vs Variance bar chart for all 3 models
ax = axes[0, 0]
models = ["Win Prob", "Margin", "Proxy"]
keys = ["win_prob", "margin", "proxy"]
bias_vals = [results[k]["bias_sq"] for k in keys]
var_vals = [results[k]["variance"] for k in keys]
x = np.arange(len(models))
ax.bar(x - 0.15, bias_vals, 0.3, label="Bias²", color="#ef4444", alpha=0.8)
ax.bar(x + 0.15, var_vals, 0.3, label="Variance", color="#3b82f6", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(models, color="white")
ax.legend(facecolor="#16213e", edgecolor="#334155", labelcolor="white")
ax.set_title("Bias² vs Variance by Model", fontsize=13, fontweight="bold")
ax.set_ylabel("Error Component")

# Plot 2: Bias-Variance by game stage
if game_progress is not None and stage_results:
    ax = axes[0, 1]
    stage_labels = [s["stage"] for s in stage_results]
    stage_bias = [s["bias_sq"] for s in stage_results]
    stage_var = [s["variance"] for s in stage_results]
    x = np.arange(len(stage_labels))
    ax.bar(x - 0.15, stage_bias, 0.3, label="Bias²", color="#ef4444", alpha=0.8)
    ax.bar(x + 0.15, stage_var, 0.3, label="Variance", color="#3b82f6", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.split("(")[0].strip() for s in stage_labels], color="white", fontsize=9)
    ax.legend(facecolor="#16213e", edgecolor="#334155", labelcolor="white")
    ax.set_title("Bias² vs Variance by Game Stage", fontsize=13, fontweight="bold")

# Plot 3: Variance heatmap (prediction vs game_progress)
ax = axes[1, 0]
if game_progress is not None:
    valid_mask = results["win_prob"]["valid_mask"]
    gp = game_progress[valid_mask]
    mp = mean_p[valid_mask]
    vp = results["win_prob"]["var_preds"][valid_mask]

    # Only plot where we have valid data
    finite = np.isfinite(gp) & np.isfinite(mp) & np.isfinite(vp)
    scatter = ax.scatter(gp[finite], mp[finite], c=vp[finite], cmap="YlOrRd",
                         s=8, alpha=0.5, vmin=0, vmax=np.percentile(vp[finite], 95))
    plt.colorbar(scatter, ax=ax, label="Variance")
    ax.set_xlabel("Game Progress")
    ax.set_ylabel("Mean Predicted P(home wins)")
    ax.set_title("Prediction Variance Map", fontsize=13, fontweight="bold")

# Plot 4: Bias² distribution
ax = axes[1, 1]
bias_vals_all = results["win_prob"]["bias_sq_per_sample"][results["win_prob"]["valid_mask"]]
bias_vals_all = bias_vals_all[np.isfinite(bias_vals_all)]
ax.hist(bias_vals_all, bins=50, color="#a78bfa", edgecolor="white", linewidth=0.3, alpha=0.8)
ax.axvline(x=np.mean(bias_vals_all), color="#facc15", linestyle="--", label=f"Mean: {np.mean(bias_vals_all):.4f}")
ax.set_xlabel("Bias² per sample")
ax.set_ylabel("Count")
ax.set_title("Distribution of Bias² (Win Prob)", fontsize=13, fontweight="bold")
ax.legend(facecolor="#16213e", edgecolor="#334155", labelcolor="white")

plt.suptitle("Bias-Variance Decomposition — Courtside Alpha v4",
             fontsize=15, fontweight="bold", color="white", y=1.02)
plt.tight_layout()
plt.savefig("bias_variance_plots.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
print(f"  Saved: bias_variance_plots.png")

print(f"\n{'=' * 70}")
print(f"  DONE")
print(f"{'=' * 70}")
