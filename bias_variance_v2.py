"""
Bias-Variance Decomposition v2 — CORRECTED
============================================
Two proper decompositions:

1. Domingos (2000) 0-1 loss decomposition:
   - Main prediction = mode of predicted CLASS across bootstraps
   - Bias = P(main_prediction ≠ y_true)
   - Variance = P(prediction ≠ main_prediction)
   - Noise = P(y_true ≠ Bayes-optimal)

2. Murphy (1973) Brier Score decomposition:
   - Reliability = calibration error (are predicted probs accurate?)
   - Resolution = discrimination (can model separate winners from losers?)
   - Uncertainty = base rate noise ō(1-ō) — irreducible

The v1 decomposition was WRONG because (mean_prob - binary_outcome)²
conflates systematic error with irreducible Bayes noise.
"""

import sys, warnings, os
import numpy as np
import xgboost as xgb
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("  BIAS-VARIANCE DECOMPOSITION v2 (CORRECTED)")
print("=" * 70)

# ── Load data (same as v1) ───────────────────────────────────────────────
print("\n[1/6] Loading training data...")

from model_v4 import (
    load_all_data, build_rolling_team_features, build_team_profiles,
    extract_game_snapshots, enrich_snapshots_with_boxscore,
    damp_live_features, drop_mismatched_features, merge_all_features,
    compute_sample_weights, get_pregame_features, get_live_features,
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
if os.path.exists("live_observations.sqlite"):
    df = merge_live_observations(df)

pregame_feats = get_pregame_features(df)
live_feats = get_live_features(df)
y_true = df["HOME_WON"].values.astype(int)
base_rate = y_true.mean()

print(f"  Data: {len(df)} rows, {df['GAME_ID'].nunique()} games")
print(f"  Features: {len(live_feats)} live + {len(pregame_feats)} pregame")
print(f"  Base rate P(home wins): {base_rate:.3f}")

# ── Bootstrap ────────────────────────────────────────────────────────────
N_BOOTSTRAPS = 20
BOOTSTRAP_FRAC = 0.8
unique_games = df["GAME_ID"].unique()
n_games = len(unique_games)
n_sample = int(n_games * BOOTSTRAP_FRAC)

print(f"\n[2/6] Running {N_BOOTSTRAPS} bootstrap iterations...")

all_probs = np.full((N_BOOTSTRAPS, len(df)), np.nan)  # predicted probabilities
all_classes = np.full((N_BOOTSTRAPS, len(df)), np.nan)  # predicted classes (0/1)

for b in range(N_BOOTSTRAPS):
    rng = np.random.RandomState(100 + b)
    boot_games = rng.choice(unique_games, size=n_sample, replace=True)
    boot_set = set(boot_games)
    oob_set = set(unique_games) - boot_set

    train_mask = df["GAME_ID"].isin(boot_set)
    oob_mask = df["GAME_ID"].isin(oob_set)
    if oob_mask.sum() == 0:
        continue

    X_train = df.loc[train_mask, live_feats].values
    y_train = y_true[train_mask]
    w_train = df.loc[train_mask, "SAMPLE_WEIGHT"].values
    X_oob = df.loc[oob_mask, live_feats].values

    model = xgb.XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.5,
        min_child_weight=10, reg_alpha=0.2, reg_lambda=1.5, gamma=0.1,
        random_state=100 + b, verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    oob_idx = np.where(oob_mask.values)[0]
    probs = model.predict_proba(X_oob)[:, 1]
    all_probs[b, oob_idx] = probs
    all_classes[b, oob_idx] = (probs >= 0.5).astype(int)

    print(f"  Bootstrap {b+1}/{N_BOOTSTRAPS}: "
          f"train={train_mask.sum()}, OOB={oob_mask.sum()} "
          f"({len(oob_set)} games)")

# ── Decomposition 1: Domingos 0-1 Loss ───────────────────────────────────
print(f"\n[3/6] Domingos (2000) 0-1 Loss Decomposition...")
print(f"{'=' * 70}")

min_oob = 3
counts = np.sum(~np.isnan(all_classes), axis=0)
valid = counts >= min_oob

# Main prediction = mode of predicted classes
from scipy import stats
main_pred = np.full(len(df), np.nan)
variance_01 = np.full(len(df), np.nan)

for i in range(len(df)):
    if not valid[i]:
        continue
    preds_i = all_classes[~np.isnan(all_classes[:, i]), i].astype(int)
    mode_result = stats.mode(preds_i, keepdims=True)
    main_pred[i] = mode_result.mode[0]
    # Variance = fraction of predictions that disagree with mode
    variance_01[i] = np.mean(preds_i != main_pred[i])

v = valid
bias_01 = (main_pred[v] != y_true[v]).astype(float)
var_01 = variance_01[v]

# Noise: for each sample, P(y ≠ Bayes optimal) — estimated from OOB agreement
# In practice, for binary classification: noise ≈ min(p, 1-p) where p = E[y|x]
# We approximate p with mean predicted probability
mean_prob = np.nanmean(all_probs[:, :], axis=0)
noise_01 = np.minimum(mean_prob[v], 1 - mean_prob[v])

print(f"\n  Samples used: {v.sum()} / {len(df)}")
print(f"  Avg OOB appearances: {counts[v].mean():.1f}")
print(f"")
print(f"  ┌─────────────────────────────────────┐")
print(f"  │  0-1 LOSS DECOMPOSITION             │")
print(f"  ├─────────────────────────────────────┤")
print(f"  │  Bias  (main≠true):  {bias_01.mean():.4f}          │")
print(f"  │  Variance (disagree): {var_01.mean():.4f}          │")
print(f"  │  Noise (Bayes err):   {noise_01.mean():.4f}          │")
print(f"  │  Total 0-1 loss:      {(bias_01 + var_01).mean():.4f}          │")
print(f"  ├─────────────────────────────────────┤")
b_ratio = bias_01.mean() / max(var_01.mean(), 1e-10)
print(f"  │  Bias/Variance ratio: {b_ratio:.2f}x           │")

if bias_01.mean() > var_01.mean() * 2:
    diagnosis = "HIGH BIAS (underfitting)"
elif var_01.mean() > bias_01.mean() * 2:
    diagnosis = "HIGH VARIANCE (overfitting)"
else:
    diagnosis = "BALANCED"
print(f"  │  Diagnosis: {diagnosis:<24}│")
print(f"  └─────────────────────────────────────┘")

# ── Decomposition 2: Brier Score (Murphy 1973) ───────────────────────────
print(f"\n[4/6] Murphy (1973) Brier Score Decomposition...")
print(f"{'=' * 70}")

mean_probs_valid = mean_prob[v]
y_valid = y_true[v]
N = len(y_valid)

# Brier score of mean predictions
brier = np.mean((mean_probs_valid - y_valid) ** 2)

# Bin predictions for reliability/resolution
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
reliability = 0.0
resolution = 0.0
bin_stats = []

for k in range(n_bins):
    mask = (mean_probs_valid >= bin_edges[k]) & (mean_probs_valid < bin_edges[k+1])
    if k == n_bins - 1:  # include right edge
        mask = (mean_probs_valid >= bin_edges[k]) & (mean_probs_valid <= bin_edges[k+1])
    n_k = mask.sum()
    if n_k == 0:
        continue
    f_k = mean_probs_valid[mask].mean()  # mean prediction in bin
    o_k = y_valid[mask].mean()           # observed frequency in bin
    reliability += (n_k / N) * (f_k - o_k) ** 2
    resolution += (n_k / N) * (o_k - base_rate) ** 2
    bin_stats.append({
        "bin": f"[{bin_edges[k]:.1f}, {bin_edges[k+1]:.1f})",
        "n": n_k, "pred_mean": f_k, "obs_freq": o_k,
        "cal_error": abs(f_k - o_k),
    })

uncertainty = base_rate * (1 - base_rate)

print(f"\n  ┌──────────────────────────────────────────────┐")
print(f"  │  BRIER SCORE DECOMPOSITION                   │")
print(f"  ├──────────────────────────────────────────────┤")
print(f"  │  Brier Score:      {brier:.6f}                  │")
print(f"  │                                              │")
print(f"  │  Reliability:      {reliability:.6f}  (calibration err) │")
print(f"  │  Resolution:       {resolution:.6f}  (discrimination)  │")
print(f"  │  Uncertainty:      {uncertainty:.6f}  (irreducible)     │")
print(f"  │                                              │")
print(f"  │  Check: REL - RES + UNC = {reliability - resolution + uncertainty:.6f}     │")
print(f"  │         Brier Score     = {brier:.6f}              │")
print(f"  ├──────────────────────────────────────────────┤")

# Interpretation
skill = 1 - brier / uncertainty
print(f"  │  Brier Skill Score:  {skill:.4f}                   │")
print(f"  │  (1.0 = perfect, 0.0 = no skill, <0 = worse) │")
print(f"  │                                              │")

if reliability > resolution:
    print(f"  │  PROBLEM: Calibration error > discrimination │")
    print(f"  │  The model would improve more from better   │")
    print(f"  │  calibration than from better features.      │")
elif resolution > reliability * 5:
    print(f"  │  GOOD: Strong discrimination, low cal error  │")
    print(f"  │  The model separates outcomes well.          │")
else:
    print(f"  │  MIXED: Both calibration and discrimination  │")
    print(f"  │  have room for improvement.                  │")

print(f"  └──────────────────────────────────────────────┘")

print(f"\n  Calibration by bin:")
print(f"  {'Bin':<15} {'N':>6} {'Pred':>8} {'Actual':>8} {'|Error|':>8}")
print(f"  {'─' * 50}")
for s in bin_stats:
    marker = " ⚠" if s["cal_error"] > 0.10 else ""
    print(f"  {s['bin']:<15} {s['n']:>6} {s['pred_mean']:>8.3f} {s['obs_freq']:>8.3f} {s['cal_error']:>8.3f}{marker}")

# ── By game stage ────────────────────────────────────────────────────────
print(f"\n[5/6] Decomposition by game stage...")
print(f"{'=' * 70}")

game_progress = df["GAME_PROGRESS"].values if "GAME_PROGRESS" in df.columns else None

if game_progress is not None:
    stages = [
        ("Q1 (0-25%)", 0.0, 0.25),
        ("Q2 (25-50%)", 0.25, 0.50),
        ("Q3 (50-75%)", 0.50, 0.75),
        ("Q4 early (75-90%)", 0.75, 0.90),
        ("Q4 late (90-100%)", 0.90, 1.01),
    ]

    print(f"\n  DOMINGOS 0-1 LOSS BY STAGE:")
    print(f"  {'Stage':<20} {'Bias':>8} {'Variance':>10} {'Noise':>8} {'B/V':>6} {'Diagnosis'}")
    print(f"  {'─' * 75}")

    stage_data_01 = []
    for label, lo, hi in stages:
        stage_mask_full = (game_progress >= lo) & (game_progress < hi)
        stage_mask = stage_mask_full[v]
        if stage_mask.sum() < 20:
            continue
        b = bias_01[stage_mask].mean()
        va = var_01[stage_mask].mean()
        n = noise_01[stage_mask].mean()
        ratio = b / max(va, 1e-10)
        diag = "HIGH BIAS" if b > va * 2 else ("HIGH VAR" if va > b * 2 else "balanced")
        print(f"  {label:<20} {b:>8.4f} {va:>10.4f} {n:>8.4f} {ratio:>6.1f}x {diag}")
        stage_data_01.append({"label": label, "bias": b, "var": va, "noise": n})

    print(f"\n  BRIER DECOMPOSITION BY STAGE:")
    print(f"  {'Stage':<20} {'Brier':>8} {'Reliab':>8} {'Resol':>8} {'Uncert':>8} {'Skill':>8}")
    print(f"  {'─' * 72}")

    stage_data_brier = []
    for label, lo, hi in stages:
        stage_mask_full = (game_progress >= lo) & (game_progress < hi)
        stage_v = stage_mask_full[v]
        if stage_v.sum() < 20:
            continue

        mp = mean_probs_valid[stage_v]
        yv = y_valid[stage_v]
        br_local = base_rate  # use overall base rate

        brier_s = np.mean((mp - yv) ** 2)
        unc_s = br_local * (1 - br_local)

        # Local reliability/resolution
        rel_s = 0.0
        res_s = 0.0
        n_s = len(yv)
        for k in range(n_bins):
            m = (mp >= bin_edges[k]) & (mp < bin_edges[k+1])
            if k == n_bins - 1:
                m = (mp >= bin_edges[k]) & (mp <= bin_edges[k+1])
            nk = m.sum()
            if nk == 0:
                continue
            fk = mp[m].mean()
            ok = yv[m].mean()
            rel_s += (nk / n_s) * (fk - ok) ** 2
            res_s += (nk / n_s) * (ok - br_local) ** 2

        skill_s = 1 - brier_s / unc_s
        print(f"  {label:<20} {brier_s:>8.4f} {rel_s:>8.4f} {res_s:>8.4f} {unc_s:>8.4f} {skill_s:>8.4f}")
        stage_data_brier.append({
            "label": label, "brier": brier_s, "rel": rel_s,
            "res": res_s, "unc": unc_s, "skill": skill_s
        })

# ── Variance of predicted probabilities (the volatility question) ────────
print(f"\n[6/6] Prediction stability analysis...")
print(f"{'=' * 70}")

prob_var = np.full(len(df), np.nan)
prob_range = np.full(len(df), np.nan)
for i in range(len(df)):
    if not valid[i]:
        continue
    p_i = all_probs[~np.isnan(all_probs[:, i]), i]
    prob_var[i] = p_i.var()
    prob_range[i] = p_i.max() - p_i.min()

pv = prob_var[v]
pr = prob_range[v]

print(f"\n  How much does the predicted probability change")
print(f"  when trained on different 80% subsets of games?")
print(f"")
print(f"  Mean prediction variance:  {np.nanmean(pv):.6f} (σ² of P(home) across bootstraps)")
print(f"  Mean prediction std dev:   {np.sqrt(np.nanmean(pv)):.4f}")
print(f"  Mean prediction range:     {np.nanmean(pr):.4f} (max - min across bootstraps)")
print(f"  Median prediction range:   {np.nanmedian(pr):.4f}")
print(f"")

if game_progress is not None:
    print(f"  Prediction stability by game stage:")
    print(f"  {'Stage':<20} {'Mean σ':>10} {'Mean range':>12} {'Interpretation'}")
    print(f"  {'─' * 65}")
    for label, lo, hi in stages:
        sm = (game_progress >= lo) & (game_progress < hi)
        sm_v = sm[v]
        if sm_v.sum() < 20:
            continue
        s = np.sqrt(np.nanmean(pv[sm_v]))
        r = np.nanmean(pr[sm_v])
        interp = "unstable" if s > 0.15 else ("moderate" if s > 0.08 else "stable")
        print(f"  {label:<20} {s:>10.4f} {r:>12.4f} {interp}")

print(f"\n  Interpretation:")
print(f"  σ < 0.05  = very stable (model barely changes with different training data)")
print(f"  σ 0.05-0.10 = moderate (normal for this setup)")
print(f"  σ > 0.15  = unstable (model is overfitting to specific training games)")

# ── Generate plots ───────────────────────────────────────────────────────
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

# Plot 1: Brier decomposition bar chart
ax = axes[0, 0]
components = ["Reliability\n(cal error)", "Resolution\n(discrimination)", "Uncertainty\n(irreducible)"]
values = [reliability, resolution, uncertainty]
colors_bar = ["#ef4444", "#22c55e", "#6b7280"]
bars = ax.bar(components, values, color=colors_bar, edgecolor="white", linewidth=0.5, alpha=0.85)
ax.set_title(f"Brier Decomposition (BSS={skill:.3f})", fontsize=13, fontweight="bold")
ax.set_ylabel("Error component", color="white")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", color="white", fontsize=10)

# Plot 2: Calibration curve (pred vs actual)
ax = axes[0, 1]
pred_means = [s["pred_mean"] for s in bin_stats]
obs_freqs = [s["obs_freq"] for s in bin_stats]
sizes = [s["n"] for s in bin_stats]
ax.plot([0, 1], [0, 1], "--", color="#6b7280", alpha=0.5, label="Perfect calibration")
ax.scatter(pred_means, obs_freqs, s=[max(20, n/10) for n in sizes],
           c="#3b82f6", edgecolors="white", linewidth=0.5, zorder=5)
for pm, of, s in zip(pred_means, obs_freqs, bin_stats):
    if abs(pm - of) > 0.1:
        ax.annotate(f"n={s['n']}", (pm, of), textcoords="offset points",
                   xytext=(5, 5), fontsize=7, color="#94a3b8")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Calibration Curve", fontsize=13, fontweight="bold")
ax.legend(facecolor="#16213e", edgecolor="#334155", labelcolor="white")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# Plot 3: Domingos decomposition by stage
if stage_data_01:
    ax = axes[1, 0]
    labels = [s["label"].split("(")[0].strip() for s in stage_data_01]
    x = np.arange(len(labels))
    bias_v = [s["bias"] for s in stage_data_01]
    var_v = [s["var"] for s in stage_data_01]
    noise_v = [s["noise"] for s in stage_data_01]
    ax.bar(x - 0.2, bias_v, 0.2, label="Bias", color="#ef4444", alpha=0.8)
    ax.bar(x, var_v, 0.2, label="Variance", color="#3b82f6", alpha=0.8)
    ax.bar(x + 0.2, noise_v, 0.2, label="Noise", color="#6b7280", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.legend(facecolor="#16213e", edgecolor="#334155", labelcolor="white", fontsize=8)
    ax.set_title("0-1 Loss Decomposition by Stage", fontsize=13, fontweight="bold")

# Plot 4: Prediction range distribution
ax = axes[1, 1]
ax.hist(pr[np.isfinite(pr)], bins=50, color="#a78bfa", edgecolor="white",
        linewidth=0.3, alpha=0.8)
ax.axvline(np.nanmean(pr), color="#facc15", linestyle="--",
           label=f"Mean range: {np.nanmean(pr):.3f}")
ax.set_xlabel("Prediction range (max - min across bootstraps)")
ax.set_ylabel("Count")
ax.set_title("Prediction Stability Distribution", fontsize=13, fontweight="bold")
ax.legend(facecolor="#16213e", edgecolor="#334155", labelcolor="white")

plt.suptitle("Bias-Variance Decomposition v2 (Corrected) — Courtside Alpha v4",
             fontsize=14, fontweight="bold", color="white", y=1.02)
plt.tight_layout()
plt.savefig("bias_variance_v2_plots.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
print(f"\n  Saved: bias_variance_v2_plots.png")

print(f"\n{'=' * 70}")
print(f"  DONE — v1 was wrong, this decomposition properly separates")
print(f"  calibration error, discrimination, and irreducible noise.")
print(f"{'=' * 70}")
