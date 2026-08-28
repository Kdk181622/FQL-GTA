"""Formal 2^3 factorial effect analysis for the ablation study: main effects of
Gaussian membership (G), eligibility traces (T), and adaptive epsilon (A), all
pairwise interactions, and the three-way interaction, fit by ordinary least
squares on the per-seed ablation data and tested via a Type-II ANOVA. This
answers, with an actual statistical test rather than a descriptive label,
whether the "synergy" observed in the raw ablation bar chart is a real
interaction effect or just three additive main effects that happen to look
large together.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "Results"

# variant -> (G, T, A) in -1/+1 coding
VARIANT_CODING = {
    "baseline": (-1, -1, -1),
    "epsilon_only": (-1, -1, 1),
    "eligibility_only": (-1, 1, -1),
    "no_gaussian": (-1, 1, 1),
    "gaussian_only": (1, -1, -1),
    "no_eligibility": (1, -1, 1),
    "gaussian_eligibility": (1, 1, -1),
    "full_gta": (1, 1, 1),
}


def run():
    df = pd.read_csv(RESULTS_DIR / "ablation_per_seed.csv")
    df["G"] = df["variant"].map(lambda v: VARIANT_CODING[v][0])
    df["T"] = df["variant"].map(lambda v: VARIANT_CODING[v][1])
    df["A"] = df["variant"].map(lambda v: VARIANT_CODING[v][2])

    # Fit the full factorial model on trailing-50 return (the "solved-ness" metric).
    model = smf.ols("trailing50_return ~ G * T * A", data=df).fit()
    anova = anova_lm(model, typ=2)

    # Yates-style effect magnitudes: mean response at +1 minus mean response at -1,
    # for each main effect and each interaction contrast (product-of-signs coding).
    df["GT"] = df["G"] * df["T"]
    df["GA"] = df["G"] * df["A"]
    df["TA"] = df["T"] * df["A"]
    df["GTA"] = df["G"] * df["T"] * df["A"]

    effects = {}
    for term in ["G", "T", "A", "GT", "GA", "TA", "GTA"]:
        hi = df.loc[df[term] == 1, "trailing50_return"].mean()
        lo = df.loc[df[term] == -1, "trailing50_return"].mean()
        effects[term] = float(hi - lo)

    term_names = {"G": "Gaussian MF (main)", "T": "Eligibility trace (main)",
                  "A": "Adaptive epsilon (main)", "GT": "Gaussian x Eligibility",
                  "GA": "Gaussian x Adaptive-eps", "TA": "Eligibility x Adaptive-eps",
                  "GTA": "Gaussian x Eligibility x Adaptive-eps"}

    anova_rows = []
    # statsmodels ANOVA index uses "G", "T", "A", "G:T", "G:A", "T:A", "G:T:A", "Residual"
    anova_key_map = {"G": "G", "T": "T", "A": "A", "GT": "G:T", "GA": "G:A",
                      "TA": "T:A", "GTA": "G:T:A"}
    for term, label in term_names.items():
        akey = anova_key_map[term]
        row = anova.loc[akey]
        anova_rows.append({
            "term": term, "label": label, "effect_magnitude": effects[term],
            "sum_sq": float(row["sum_sq"]), "df": float(row["df"]),
            "F": float(row["F"]), "p_value": float(row["PR(>F)"]),
            "significant_0.05": bool(row["PR(>F)"] < 0.05),
        })

    out = pd.DataFrame(anova_rows)
    out.to_csv(RESULTS_DIR / "factorial_effects.csv", index=False)

    residual_row = anova.loc["Residual"]
    meta = {
        "r_squared": float(model.rsquared),
        "residual_sum_sq": float(residual_row["sum_sq"]),
        "residual_df": float(residual_row["df"]),
        "n_observations": int(len(df)),
        "response_variable": "trailing50_return",
    }
    with open(RESULTS_DIR / "factorial_effects_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(out.to_string(index=False))
    print()
    print(json.dumps(meta, indent=2))
    return out, meta


if __name__ == "__main__":
    run()
