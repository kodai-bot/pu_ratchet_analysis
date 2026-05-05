"""
Supplementary code: reproducible analysis of procedural-unit (PU) acquisition
across 3.3 Myr of the hominin lithic record.

Companion to:
    Foley, D. (2026). "From Typology to Physics: A Physical Definition of
    Technology Tested Across 3.3 Million Years of Hominin Evolution."
    Submitted to the Journal of Anthropological Sciences.

Reproduces the empirical claims in §4 (quantitative tests) of the manuscript:

    1. Cumulative procedural diversity follows an exponential trajectory
       across 3.3 Myr (linear vs. exponential model comparison).
    2. The acceleration is statistically significant under a uniform null
       (permutation test on first-appearance times).
    3. Per-assemblage chain complexity rises across epochs (mean and
       maximum PU count per assemblage by period).
    4. Repertoire is irreversible: no PU disappears after first appearance.

Usage:
    python supplementary_pu_acceleration_analysis.py

Inputs:
    procedural_units_obfus.csv  --  155 dated assemblages, 33 PU columns
                                    (Paige & Perreault 2023; Zenodo
                                    doi:10.5281/zenodo.7847839)

Outputs:
    Console output with all reported statistics
    fig_pu_acceleration.pdf -- two-panel figure (cumulative + per-assemblage)
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = "procedural_units_obfus.csv"
FIG_PATH = "fig_pu_acceleration.pdf"
RNG_SEED = 42
N_PERMUTATIONS = 10_000

# Period bins (in ka, midpoint age)
PERIODS = [
    ("Pre-Oldowan",      3300, 2700),
    ("Oldowan",          2700, 1700),
    ("Early Acheulean",  1700, 1000),
    ("Middle Acheulean", 1000,  500),
    ("Late Acheulean",    500,  300),
    ("MSA",               300,   50),
    ("UP+",                50,    0),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """Load the procedural-units CSV (cp1252 encoded) and add midpoint age."""
    df = pd.read_csv(path, encoding="cp1252")
    df["age_mid"] = (df["KA.young"] + df["KA.old"]) / 2.0
    return df


def get_pu_columns(df: pd.DataFrame) -> list[str]:
    """Return the procedural-unit column names (those starting 'PU.')."""
    return [c for c in df.columns if c.startswith("PU.")]


def assign_period(age_ka: float) -> str:
    """Assign an assemblage to a period based on its midpoint age."""
    for name, old, young in PERIODS:
        if young <= age_ka <= old:
            return name
    return "Other"


# ---------------------------------------------------------------------------
# Analysis 1: cumulative trajectory (linear vs. exponential)
# ---------------------------------------------------------------------------
def first_appearance_table(df: pd.DataFrame, pu_cols: list[str]) -> pd.DataFrame:
    """For each active PU, return its first appearance age (oldest assemblage
    in which presence == 1)."""
    rows = []
    for pu in pu_cols:
        present = df.loc[df[pu] == 1, "age_mid"]
        if len(present) > 0:
            rows.append({"pu": pu, "first_appearance_ka": present.max()})
    fa = pd.DataFrame(rows).sort_values("first_appearance_ka", ascending=False)
    fa = fa.reset_index(drop=True)
    fa["cumulative_count"] = np.arange(1, len(fa) + 1)
    return fa


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = a*x + b. Returns slope, intercept, R^2, RSS, predictions."""
    slope, intercept, r, _, _ = linregress(x, y)
    yhat = slope * x + intercept
    rss = float(np.sum((y - yhat) ** 2))
    return {"slope": slope, "intercept": intercept,
            "r2": r ** 2, "rss": rss, "yhat": yhat}


def fit_exponential(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = A * exp(k * x). Returns A, k, doubling time, R^2, RSS, predictions.
    x is time in Ma counted forward from oldest assemblage so k > 0."""
    def model(t, A, k):
        return A * np.exp(k * t)
    p0 = (max(y[0], 1.0), 0.001)
    popt, _ = curve_fit(model, x, y, p0=p0, maxfev=20000)
    A, k = popt
    yhat = model(x, A, k)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    doubling_time = np.log(2.0) / k if k > 0 else np.nan
    return {"A": A, "k": k, "doubling_time": doubling_time,
            "r2": r2, "rss": ss_res, "yhat": yhat}


# ---------------------------------------------------------------------------
# Analysis 2: permutation test on rate ratio
# ---------------------------------------------------------------------------
def permutation_rate_test(first_appearance_ka: np.ndarray,
                          early_window_ka: tuple[float, float],
                          late_window_ka: tuple[float, float],
                          n: int = N_PERMUTATIONS,
                          seed: int = RNG_SEED) -> dict:
    """Permutation test: under uniform first-appearance times across the
    observed range, what is the probability of seeing the observed
    rate ratio (PUs per Myr in late window / PUs per Myr in early window)?"""
    rng = np.random.default_rng(seed)
    fa = np.asarray(first_appearance_ka, dtype=float)
    t_old, t_young = float(np.max(fa)), float(np.min(fa))

    def rate_ratio(times: np.ndarray) -> float:
        early = ((times <= early_window_ka[0]) & (times > early_window_ka[1])).sum()
        late = ((times <= late_window_ka[0]) & (times > late_window_ka[1])).sum()
        early_rate = early / ((early_window_ka[0] - early_window_ka[1]) / 1000.0)
        late_rate = late / ((late_window_ka[0] - late_window_ka[1]) / 1000.0)
        if early_rate == 0:
            return np.inf if late_rate > 0 else np.nan
        return late_rate / early_rate

    observed = rate_ratio(fa)

    # Uniform null: redistribute first-appearance times uniformly between
    # observed oldest and youngest first-appearance.
    null_ratios = np.empty(n)
    for i in range(n):
        sim = rng.uniform(t_young, t_old, size=fa.size)
        null_ratios[i] = rate_ratio(sim)

    finite = null_ratios[np.isfinite(null_ratios)]
    p = float(np.mean(finite >= observed)) if finite.size else np.nan
    return {"observed_ratio": float(observed),
            "p_value": p,
            "null_mean": float(np.mean(finite)),
            "n_permutations": int(finite.size)}


# ---------------------------------------------------------------------------
# Analysis 3: per-assemblage complexity by period
# ---------------------------------------------------------------------------
def per_assemblage_complexity(df: pd.DataFrame, pu_cols: list[str]) -> pd.DataFrame:
    """Mean and maximum PU count per assemblage, by period."""
    df = df.copy()
    df["pu_count"] = df[pu_cols].sum(axis=1)
    df["period"] = df["age_mid"].apply(assign_period)
    rows = []
    for name, _, _ in PERIODS:
        sub = df[df["period"] == name]
        if len(sub) == 0:
            continue
        rows.append({
            "period": name,
            "n_assemblages": int(len(sub)),
            "mean_pu": float(sub["pu_count"].mean()),
            "std_pu": float(sub["pu_count"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "max_pu": int(sub["pu_count"].max()),
            "cv": (float(sub["pu_count"].std(ddof=1) / sub["pu_count"].mean())
                   if len(sub) > 1 and sub["pu_count"].mean() > 0 else 0.0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 4: irreversibility check
# ---------------------------------------------------------------------------
def irreversibility_check(df: pd.DataFrame, pu_cols: list[str]) -> dict:
    """For each PU, check whether it disappears from the record after first
    appearance. A 'loss' is defined as no occurrence in any assemblage younger
    than 100 ka after a first appearance older than 100 ka."""
    losses = []
    sorted_df = df.sort_values("age_mid", ascending=False)
    for pu in pu_cols:
        present_ages = sorted_df.loc[sorted_df[pu] == 1, "age_mid"].values
        if len(present_ages) == 0:
            continue
        first = present_ages.max()
        last = present_ages.min()
        # If first appearance is > 100 ka and there are no occurrences
        # within the most recent 100 ka, treat as potential loss.
        if first > 100 and last > 100:
            losses.append({"pu": pu, "first_ka": first, "last_ka": last})
    return {"n_active_pus": int(sum(df[c].sum() > 0 for c in pu_cols)),
            "n_losses": len(losses),
            "losses": losses}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figure(fa: pd.DataFrame, lin: dict, exp: dict,
                complexity: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: cumulative
    ax = axes[0]
    age_ma = fa["first_appearance_ka"].values / 1000.0
    ax.scatter(age_ma, fa["cumulative_count"], s=22, color="black",
               zorder=3, label="observed")
    order = np.argsort(age_ma)
    ax.plot(age_ma[order], lin["yhat"][order], "--", color="grey",
            label=f"linear (R²={lin['r2']:.3f})")
    ax.plot(age_ma[order], exp["yhat"][order], "-", color="firebrick",
            linewidth=1.5,
            label=f"exponential (R²={exp['r2']:.3f}, t½≈{exp['doubling_time']*1000:.0f} kyr)")
    ax.invert_xaxis()
    ax.set_xlabel("age (Ma BP)")
    ax.set_ylabel("cumulative new PUs")
    ax.set_title("A. Cumulative procedural diversity")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel B: per-assemblage
    ax = axes[1]
    if len(complexity) > 0:
        pos = np.arange(len(complexity))
        ax.bar(pos, complexity["mean_pu"], yerr=complexity["std_pu"],
               color="steelblue", alpha=0.75, capsize=3, label="mean ± 1 SD")
        ax.plot(pos, complexity["max_pu"], "o-", color="firebrick",
                markersize=5, label="max")
        ax.set_xticks(pos)
        ax.set_xticklabels(complexity["period"], rotation=30, ha="right",
                           fontsize=8)
        ax.set_ylabel("PUs per assemblage")
        ax.set_title("B. Per-assemblage chain complexity by period")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nFigure written to: {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("Supplementary analysis: PU acquisition across 3.3 Myr")
    print("=" * 72)

    df = load_dataset(CSV_PATH)
    pu_cols = get_pu_columns(df)
    print(f"\nLoaded {len(df)} assemblages with {len(pu_cols)} PU columns.")
    print(f"Age range: {df['age_mid'].min():.1f}--{df['age_mid'].max():.1f} ka.")

    # ---------------------------------------------------------------------
    # 1. Cumulative trajectory
    # ---------------------------------------------------------------------
    fa = first_appearance_table(df, pu_cols)
    age_ma = fa["first_appearance_ka"].values / 1000.0
    # Time forward from oldest first-appearance for the exponential fit
    t = age_ma.max() - age_ma
    y = fa["cumulative_count"].values.astype(float)

    lin = fit_linear(t, y)
    exp = fit_exponential(t, y)

    print("\n--- 1. Cumulative trajectory ---")
    print(f"   Active PUs:               {len(fa)}")
    print(f"   Linear fit:               R² = {lin['r2']:.3f}, RSS = {lin['rss']:.2f}")
    print(f"   Exponential fit:          R² = {exp['r2']:.3f}, RSS = {exp['rss']:.2f}")
    print(f"   RSS ratio (linear/exp):   {lin['rss']/exp['rss']:.2f}×")
    print(f"   Doubling time:            {exp['doubling_time']*1000:.0f} kyr")

    # ---------------------------------------------------------------------
    # 2. Permutation test on rate ratio
    # ---------------------------------------------------------------------
    perm = permutation_rate_test(
        first_appearance_ka=fa["first_appearance_ka"].values,
        early_window_ka=(3300.0, 500.0),
        late_window_ka=(500.0, 0.0),
        n=N_PERMUTATIONS,
        seed=RNG_SEED,
    )
    print("\n--- 2. Permutation test ---")
    print(f"   Observed rate ratio (last 500 kyr / first 2.8 Myr):"
          f" {perm['observed_ratio']:.2f}×")
    print(f"   Null mean ratio:           {perm['null_mean']:.2f}×")
    print(f"   p-value:                   {perm['p_value']:.4f}"
          f"   ({perm['n_permutations']:,} permutations)")

    # ---------------------------------------------------------------------
    # 3. Per-assemblage complexity by period
    # ---------------------------------------------------------------------
    complexity = per_assemblage_complexity(df, pu_cols)
    print("\n--- 3. Per-assemblage chain complexity ---")
    print(complexity.to_string(index=False,
                               float_format=lambda v: f"{v:.2f}"))

    # ---------------------------------------------------------------------
    # 4. Irreversibility
    # ---------------------------------------------------------------------
    irrev = irreversibility_check(df, pu_cols)
    print("\n--- 4. Irreversibility check ---")
    print(f"   Active PUs:                  {irrev['n_active_pus']}")
    print(f"   PUs lost after appearance:   {irrev['n_losses']}")
    if irrev["n_losses"] == 0:
        print("   No PU disappears from the record after first appearance.")

    # ---------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------
    make_figure(fa, lin, exp, complexity, FIG_PATH)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
