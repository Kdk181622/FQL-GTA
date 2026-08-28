"""Statistical significance tests for the journal submission.

Primary comparison (Baseline FQL vs. FQL-GTA, 30 seeds): Cohen's d, paired
Wilcoxon signed-rank test, paired t-test -- unchanged from the assignment-track
version, now run on the full 30-seed re-run.

Ablation study (8-condition 2^3 factorial, 3 seeds each): Shapiro-Wilk
normality check per condition, 10,000-resample bootstrap 95% CI per condition,
a Friedman test across all 8 conditions (paired by seed), Nemenyi post-hoc,
and Holm-corrected pairwise Wilcoxon signed-rank tests over all C(8,2)=28
pairs -- the same statistical stack used for the Assignment 2 journal
revision, applied here for consistency across both papers.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "Results"

N_BOOTSTRAP = 10000
RNG_SEED = 42


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled_std = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
                          / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled_std


def bootstrap_ci(x, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def holm_correct(pvals):
    """Step-down Holm-Bonferroni correction. Returns adjusted p-values in the
    original order of `pvals`."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def run():
    df = pd.read_csv(RESULTS_DIR / "learning_curves.csv")
    out = {}

    # -------------------------------------------------------------------
    # Primary comparison: per-seed trailing-50 return (30 seeds).
    # -------------------------------------------------------------------
    trailing = {}
    for agent in df["agent"].unique():
        sub = df[df["agent"] == agent]
        vals = []
        for seed in sorted(sub["seed"].unique()):
            returns = sub[(sub["seed"] == seed)].sort_values("episode")["return"].values
            vals.append(returns[-50:].mean())
        trailing[agent] = np.array(vals)

    base = trailing["Baseline FQL"]
    gta = trailing["Improved FQL-GTA"]
    d = cohens_d(gta, base)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(base, gta)
    ttest_stat, ttest_p = stats.ttest_rel(base, gta)
    shapiro_base = stats.shapiro(base)
    shapiro_gta = stats.shapiro(gta)
    out["primary_comparison"] = {
        "n_seeds": len(base),
        "cohens_d_trailing50_return": float(d),
        "wilcoxon_statistic": float(wilcoxon_stat),
        "wilcoxon_p_value": float(wilcoxon_p),
        "paired_ttest_statistic": float(ttest_stat),
        "paired_ttest_p_value": float(ttest_p),
        "shapiro_wilk_baseline": {"statistic": float(shapiro_base.statistic), "p_value": float(shapiro_base.pvalue)},
        "shapiro_wilk_gta": {"statistic": float(shapiro_gta.statistic), "p_value": float(shapiro_gta.pvalue)},
        "bootstrap_95ci_baseline": bootstrap_ci(base),
        "bootstrap_95ci_gta": bootstrap_ci(gta),
        "n_pairs": len(base),
    }

    # -------------------------------------------------------------------
    # Ablation study: 8-condition 2^3 factorial, 3 seeds each.
    # -------------------------------------------------------------------
    ablation_path = RESULTS_DIR / "ablation_per_seed.csv"
    if ablation_path.exists():
        abl = pd.read_csv(ablation_path)
        labels = abl["label"].unique().tolist()

        # Per-condition Shapiro-Wilk and bootstrap 95% CI, on avg_reward.
        per_condition = {}
        for label in labels:
            vals = abl.loc[abl["label"] == label, "avg_reward"].values
            sw = stats.shapiro(vals)
            per_condition[label] = {
                "n_seeds": len(vals),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
                "shapiro_wilk_statistic": float(sw.statistic),
                "shapiro_wilk_p_value": float(sw.pvalue),
                "bootstrap_95ci": bootstrap_ci(vals),
            }
        out["ablation_per_condition"] = per_condition
        out["ablation_normality_note"] = (
            "Shapiro-Wilk run on n=3 seeds per condition has very low power; "
            "treat these as descriptive, not a reliable normality gate. Bootstrap "
            "CIs and the non-parametric Friedman/Wilcoxon tests below are the "
            "primary evidence for this reason."
        )

        # One-way ANOVA retained for continuity with the assignment-track report.
        groups = [g["avg_reward"].values for _, g in abl.groupby("label")]
        f_stat, anova_p = stats.f_oneway(*groups)
        out["ablation_anova"] = {
            "f_statistic": float(f_stat), "p_value": float(anova_p),
            "n_groups": len(groups), "labels": labels,
        }

        # Friedman test: conditions as treatments, seeds as blocks. Requires a
        # seed x condition matrix with no missing cells.
        wide = abl.pivot(index="seed", columns="label", values="avg_reward")[labels]
        friedman_stat, friedman_p = stats.friedmanchisquare(*[wide[c].values for c in labels])
        out["friedman_test"] = {
            "statistic": float(friedman_stat), "p_value": float(friedman_p),
            "n_conditions": len(labels), "n_blocks_seeds": wide.shape[0],
        }

        # Nemenyi post-hoc (matrix of pairwise p-values, Friedman-consistent).
        nemenyi = sp.posthoc_nemenyi_friedman(wide.values)
        nemenyi.index = labels
        nemenyi.columns = labels
        nemenyi.to_csv(RESULTS_DIR / "ablation_nemenyi_pvalues.csv")

        # Holm-corrected pairwise Wilcoxon signed-rank over all C(8,2)=28 pairs.
        pair_rows = []
        raw_p = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = wide[labels[i]].values, wide[labels[j]].values
                try:
                    stat, p = stats.wilcoxon(a, b)
                except ValueError:
                    stat, p = np.nan, 1.0
                pair_rows.append({"condition_a": labels[i], "condition_b": labels[j],
                                   "wilcoxon_statistic": stat, "p_value_raw": p})
                raw_p.append(p)
        adj_p = holm_correct(raw_p)
        for row, p_adj in zip(pair_rows, adj_p):
            row["p_value_holm"] = float(p_adj)
            row["significant_holm_0.05"] = bool(p_adj < 0.05)
        pairwise_df = pd.DataFrame(pair_rows)
        pairwise_df.to_csv(RESULTS_DIR / "ablation_pairwise_wilcoxon_holm.csv", index=False)
        out["ablation_pairwise_holm_summary"] = {
            "n_pairs_tested": len(pair_rows),
            "n_significant_after_holm": int(pairwise_df["significant_holm_0.05"].sum()),
        }

    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
