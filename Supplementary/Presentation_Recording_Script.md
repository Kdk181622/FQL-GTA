# Recording Script — FQL-GTA (Journal Track)
**Target length:** ~5-6 minutes · **Delivery:** screen-share the deck, advance one slide per section.

> This script matches `Presentation_FQL-GTA.pptx` (13 slides). Trimmed for a tight, high-density
> delivery: every sentence carries a specific number or finding, no filler. All numbers are
> regenerated directly from `Results/*.csv|json`, so the deck, manuscript, and this script agree
> exactly.

---

**[Slide 1 — Title]**
This is a systematically evaluated, statistically validated enhancement to classical Fuzzy Q-Learning, combining Gaussian membership functions, eligibility traces, and adaptive exploration. Work under Dr. J. B. Simha.

**[Slide 2 — Two Research Questions]**
Two questions drive this paper. Does combining these three enhancements beat classical FQL, and is that gain attributable to the combination, not one component? And do CartPole-tuned hyperparameters transfer zero-shot to new environments? Both are answered with data, including the honest limitations.

**[Slide 3 — Why Fuzzy Q-Learning]**
A trained deep policy is a black box; a fuzzy controller's decisions trace to specific rules and firing strengths. That interpretability matters in safety-critical control. Classical FQL from 1998 uses triangular membership, one-step updates, and fixed exploration -- choices this study tests directly.

**[Slide 4 — Primary Comparison]**
Across thirty seeds, FQL-GTA solves CartPole -- strictly defined as still above threshold at episode 500 -- in forty percent of seeds, against zero percent for the baseline. Paired Wilcoxon test confirms this with a large effect size, Cohen's d of 2.51.

**[Slide 5 — Learning Curves]**
Eighty percent of seeds cross the solve threshold at some point; only forty percent hold it through episode 500. That gap is exactly why "solved" is scored from the final trailing window, not first crossing -- a run that peaks and collapses is not a success.

**[Slide 6 — Full Factorial Ablation]**
All eight combinations of the three enhancements were tested, not just added one at a time. No single enhancement comes close to solving the task alone; even two of three together fall short. A formal factorial effect analysis shows significant two-way interaction effects between Gaussian membership and eligibility traces, and between Gaussian membership and adaptive epsilon, with no significant three-way interaction.

**[Slide 7 — Statistics, Including the Limitation]**
Normality is rejected for both agents, so Wilcoxon leads, not a t-test -- and it's significant. For the ablation, an omnibus test across all eight conditions is significant, but at only three seeds per condition, none of the twenty-eight Holm-corrected pairwise comparisons hold up individually. That limitation is reported directly, not hidden behind the significant omnibus result.

**[Slide 8 — Hyperparameter Sensitivity]**
The tuned hyperparameters sit at pronounced performance peaks within the tested parameter ranges, not an arbitrary point on a flat surface. Widen the Gaussian membership past its tuned value and performance collapses to near-random, because every rule fires for every state and local specialisation disappears.

**[Slide 9 — Cross-Environment Transfer]**
Direct test of the second question: CartPole-tuned hyperparameters applied with no retuning, versus a cheap coordinate search on a held-out validation seed. Zero-shot transfer is weak, especially on the harder environment. Retuning -- at no extra compute cost beyond the sensitivity sweep -- recovers nearly all of the lost performance.

**[Slide 10 — Deep-RL Baselines]**
Benchmarked against a hand-rolled DQN and stable-baselines3's DQN and PPO under a comparable budget. PPO wins on final performance but at roughly fourteen times the parameters and memory. Both DQN variants actually under-perform FQL-GTA within this budget -- a budget-dependent finding, not a claim that fuzzy Q-learning beats deep RL generally.

**[Slide 11 — Interpretability, Concretely]**
One real trained-agent decision, fully traced: the state, which of the three hundred twenty-four rules actually fire, the dominant rule's firing strength and Q-values, and the resulting blended decision. Only a handful of rules fire per state -- confirmation that Gaussian membership preserves local specialisation, not uniform activation.

**[Slide 12 — Conclusion]**
Summary: the combination produces a large, statistically robust improvement, driven by significant two-way interaction effects, with no significant three-way interaction, with the ablation's power limitation reported honestly. Zero-shot transfer is weak, but cheap retuning fixes most of it. Against deep RL, this method trades some final performance for an order-of-magnitude smaller model, faster training, and full rule-level interpretability. The contribution is deliberately modest: not new ingredients, but a systematically evaluated, reproducible enhancement to a twenty-six-year-old method.

**[Slide 13 — Thank You]**
Thank you. Code, configuration, results, and the manuscript accompany this presentation.

---
*Tip: speak the numbers in slides 4, 6, 7, and 9 slowly -- they carry the paper's central findings.*
