# A Systematically Evaluated Interpretable Enhancement to Fuzzy Q-Learning (FQL-GTA)

**Author:** Karthikeyan K · MS in Artificial Intelligence & Machine Learning, REVA University
**Advisor:** Dr. J. B. Simha
**Status:** Manuscript prepared for journal submission. **Not yet submitted to any venue.**
**DOI: Not yet assigned.**

> Submission status is a strict progression: manuscript prepared -> submitted -> under review ->
> accepted -> published. This work is at the first stage. Nothing in this repository should be
> read as a claim that the manuscript has been published.

This repository accompanies a manuscript prepared for journal submission. It is a substantial
revision of an earlier MSc coursework submission, raised to journal standard: the primary
comparison is re-run at 30 independent seeds (up from 5), the ablation study is extended to a full
2^3 factorial design, a formal cross-environment transfer study replaces an informal
generalisation check, deep-RL baselines (a hand-rolled DQN, stable-baselines3's DQN and PPO) are
added for context, and every reported number is regenerated from a single frozen configuration
(`configs/default_config.json`).

## Research questions

1. Does combining Gaussian membership functions, Q(lambda) eligibility traces, and an adaptive
   epsilon schedule produce a statistically defensible improvement over classical Fuzzy
   Q-Learning (Jouffe, 1998), and is that improvement attributable to the combination rather than
   any single component?
2. Do CartPole-tuned FQL-GTA hyperparameters transfer zero-shot to other control environments, or
   is per-environment retuning necessary?

## Headline findings (reported plainly, including the less flattering ones)

- **The improvement is large and statistically well-supported.** Across 30 seeds, FQL-GTA reaches
  a final trailing-50 return of 423.1 (SD 78.7) vs. 265.7 (SD 41.2) for the baseline (paired
  Wilcoxon p ≈ 3.5e-8, Cohen's d ≈ 2.51).
- **The result is not a learning-rate artifact.** The baseline and FQL-GTA use different learning
  rates by frozen configuration (0.3 vs. 0.2). A dedicated sweep of the baseline's learning rate
  over {0.1, 0.2, 0.3, 0.5} (same 30 seeds, same 500-episode budget) finds the best baseline
  (α=0.5: 275.5 ± 28.0) still loses to FQL-GTA by a materially unchanged margin (d ≈ 2.50,
  p ≈ 4.7e-8) — see manuscript Section 6.1.
- **No single component solves the task alone, and the interaction structure is precise, not a**
  **blanket "synergy."** A formal 2^3 factorial effect analysis (OLS + Type-II ANOVA on the ablation
  data) finds all three main effects significant, a significant *positive* interaction between
  Gaussian membership and eligibility traces (they amplify each other), a significant *negative*
  interaction between Gaussian membership and adaptive epsilon (diminishing returns), and no
  significant eligibility-trace x adaptive-epsilon or three-way interaction.
- **Performance bought per unit of compute is favourable.** FQL-GTA gains +58% final return over
  the baseline at 2.6x its training time; PPO gains more (+83%) but needs 11.6x the training time
  to get there.
- **The ablation's pairwise comparisons are underpowered.** A Friedman omnibus test across the 8
  conditions is significant (p ≈ 0.011), but with only 3 seeds per condition, none of the 28
  Holm-corrected pairwise Wilcoxon comparisons reach significance individually. Reported honestly
  as a power limitation, not hidden.
- **Hyperparameters sit at pronounced performance peaks within the tested parameter ranges**,
  not an arbitrary point on a flat surface — the
  Gaussian width multiplier in particular shows a cliff: values ≥ 0.5 collapse performance to
  near-random.
- **Zero-shot transfer to MountainCar-v0 and Acrobot-v1 is weak**, especially on Acrobot-v1
  (trailing-50 ≈ -300 vs. a -100 practical bar). A lightweight, disjoint-validation-seed
  coordinate search closes most of the gap (Acrobot-v1 tuned ≈ -103, nearly at the bar).
- **Against deep-RL baselines**, PPO wins on final return but at ~14x FQL-GTA's parameter count
  and ~14x its estimated memory footprint (35.76 KB vs. 2.53 KB); both a hand-rolled DQN and
  stable-baselines3's DQN under-perform FQL-GTA within this study's training budget — reported as
  a budget-dependent finding, not a general claim that FQL-GTA beats deep RL.

## What this manuscript does NOT claim

- **Not** "FQL-GTA is superior to deep reinforcement learning." PPO reaches a higher final return
  in this study; FQL-GTA's advantage is interpretability, parameter count, and training cost, not
  raw accuracy.
- **Not** unconditional cross-environment generalisation. The multi-environment results include a
  weak zero-shot transfer to Acrobot-v1, reported directly rather than omitted.

## Repository structure

```
Assignment_1_FQL-GTA/
├── Title_Page/
│   └── Title_Page.docx                 # author name, affiliation, corresponding-author email,
│                                        # funding, acknowledgments, author contributions
├── Manuscript/
│   └── FQL-GTA_Final_Manuscript.docx   # the journal manuscript -- blinded: no author name,
│                                        # affiliation, or advisor mention anywhere in the text
├── PDF/
│   └── FQL-GTA_Final_Manuscript.pdf    # PDF export of the manuscript
├── Cover_Letter/
│   └── Cover_Letter.docx
├── Highlights/
│   └── Highlights.docx                 # Elsevier-style 3-5 bullet highlights
├── Graphical_Abstract/
│   └── Graphical_Abstract.png
├── Journal_Submission_Checklist/
│   └── Journal_Submission_Checklist.docx
├── Figures/                            # every figure in the manuscript, regenerable from Results/
├── Results/                            # every CSV/JSON artifact behind every number in the paper
├── Data/
│   ├── DATA_CARD.md                    # provenance and schema
│   └── cartpole_transitions.csv        # transition log from a trained agent (regenerable)
├── Code/                               # all model code + experiment/analysis/figure scripts
├── configs/
│   └── default_config.json             # the single frozen configuration every script reads
├── notebooks/
│   └── FQL_GTA_Study.ipynb             # end-to-end runnable notebook version
├── Supplementary/
│   ├── Presentation_FQL-GTA.pptx
│   ├── FQL-GTA_Presentation_Video.mp4
│   ├── Presentation_Recording_Script.md
│   ├── Selected_Paper_Reference.md
│   └── Paper_Critical_Analysis.docx
├── README/
│   └── README.md                       # this file
├── requirements.txt                    # pinned pip dependencies
├── environment.yml                     # conda equivalent
└── LICENSE                             # MIT (code); see note on environment/dataset in the file
```

## Reproduction

```bash
# 1. Environment
pip install -r requirements.txt
# or: conda env create -f environment.yml

# 2. Primary comparison (30 seeds x 500 episodes x 2 agents, ~25-40 min on a consumer CPU)
python Code/run_experiments.py

# 3. Full 2^3 factorial ablation (8 conditions x 3 seeds, ~5 min)
python Code/run_ablation_study.py

# 4. Statistical tests (Shapiro-Wilk, bootstrap CI, Friedman, Nemenyi, Holm-corrected Wilcoxon)
python Code/run_statistical_tests.py

# 4b. Formal 2^3 factorial effect analysis (main effects + interactions, OLS/ANOVA)
python Code/run_factorial_effects.py

# 4c. Baseline learning-rate fairness check (30 seeds x 4 alpha values, ~30 min)
python Code/run_alpha_sensitivity.py

# 5. Hyperparameter sensitivity sweep (lambda / sigma_mult / epsilon_decay, ~35 min)
python Code/run_hyperparameter_sensitivity.py

# 6. Cross-environment transfer study (zero-shot vs. tuned, MountainCar-v0 + Acrobot-v1, ~35 min)
python Code/run_cross_env_transfer.py

# 7. Deep-RL baselines (hand-rolled DQN, stable-baselines3 DQN + PPO, ~60-90 min total)
python Code/dqn_baseline.py          # smoke test
python Code/run_rl_baselines.py      # full 10-seed comparison across all 5 methods

# 8. Rule-level interpretability worked examples
python Code/run_learned_rules.py

# 9. Computational cost measurement
python Code/run_computational_cost.py

# 10. Regenerate all figures, the manuscript, the slide deck, and the submission documents
python Code/make_figures.py
python Code/make_extended_figures.py
python Code/make_figures_journal.py
python Code/make_manuscript_journal.py
python Code/make_presentation_journal.py
python Code/make_submission_docs.py
```

**Exact environment used to produce the reported numbers:** Windows 11, a consumer-grade CPU (no
GPU required). Package versions are pinned as minimums in `requirements.txt`; exact versions used
are recorded in `configs/default_config.json`. All random seeds are fixed and listed
per-experiment in that same file, so every reported number is exactly reproducible from a clean
checkout.

## Data availability statement

- **Training environments:** CartPole-v1, MountainCar-v0, and Acrobot-v1 from the Gymnasium
  library (Farama Foundation, MIT licensed). No static dataset is consumed; all training signal is
  generated online through agent-environment interaction under the fixed seeds in
  `configs/default_config.json`.
- **`Data/cartpole_transitions.csv`:** a transition-level log from a trained agent's greedy
  rollout, generated by `Code/export_dataset.py`, fully regenerable and included for offline
  inspection; see `Data/DATA_CARD.md` for schema.
- **All experiment outputs** (per-seed results, summary statistics, ablation/sensitivity/transfer
  results, statistical test outputs) are included in `Results/` in full, not only as summary
  tables in the manuscript.

## Code availability statement

The source code, configuration files, and experiment scripts used to generate every reported
result are included in full in `Code/` and are exactly reproducible via the commands above.

## Citation

If you use this code or build on this work, please cite the manuscript (citation details to be
finalised on acceptance; in the interim, cite this repository directly).

## License

Source code: MIT License (see `LICENSE`). Gymnasium environments: MIT License (Farama Foundation).

## Changelog

- **v1.3 (editorial-office corrections, applied proactively):** added a separate Title_Page/
  (corresponding-author mark, institutional email, funding, acknowledgments, author
  contributions) and blinded the manuscript body (removed author name/affiliation/advisor
  mention); added a Conflict of Interest section before the reference section; switched the
  corresponding-author email from a personal address to the institutional address across the
  title page and cover letter -- matching corrections requested by the editorial office for the
  companion FCM-PSO-ANFIS submission, applied here before this paper is submitted.
- **v1.2 (baseline learning-rate fairness check):** added a dedicated sweep of the classical
  baseline's learning rate (Section 6.1) after finding the baseline (α=0.3) and FQL-GTA (α=0.2)
  used different learning rates without this being documented; fixed a stale code comment in
  `fql_improved.py` that incorrectly claimed alpha matched the baseline; confirmed the primary
  comparison's effect size is materially unchanged against the best-tuned baseline.
- **v1.1 (submission packaging):** repository restructured into a journal-submission layout
  (Manuscript/, PDF/, Cover_Letter/, Highlights/, Graphical_Abstract/, Supplementary/,
  Journal_Submission_Checklist/); added a memory-footprint column to the deep-RL baseline
  comparison; corrected a stale hyperparameter table in the notebook.
- **v1.0 (journal revision):** 30-seed primary comparison (up from 5); full 2^3 factorial ablation
  (up from 4 leave-one-out conditions) with Shapiro-Wilk/bootstrap-CI/Friedman/Nemenyi/Holm-corrected
  statistics; hyperparameter sensitivity sweep; formal zero-shot-vs-tuned cross-environment transfer
  study (replacing an informal baseline-vs-GTA multi-environment check); deep-RL baseline comparison
  (hand-rolled DQN, stable-baselines3 DQN + PPO); rule-level interpretability worked examples; full
  reproducibility package with frozen configuration.
- **v0 (assignment coursework):** initial FQL-GTA study at 5 seeds, 4-condition ablation, informal
  multi-environment generalisation check.
