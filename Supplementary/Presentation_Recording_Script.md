# Recording Script — FQL-GTA (Journal Track)
**Target length:** ~9-10 minutes · **Delivery:** screen-share the deck, advance one slide per section.

> This script matches `Presentation_FQL-GTA.pptx` (13 slides). Every sentence carries a specific
> number or finding, no filler. All numbers are regenerated directly from `Results/*.csv|json`, so
> the deck, manuscript, and this script agree exactly.

---

**[Slide 1 — Title]**
This is a systematically evaluated, statistically validated enhancement to classical Fuzzy Q-Learning, combining Gaussian membership functions, eligibility traces, and adaptive exploration. Work under Dr. J. B. Simha. The whole point of this study is to test that combination rather than simply assume it works, and to report exactly how much of the improvement each piece is responsible for.

**[Slide 2 — Two Research Questions]**
Two questions drive this paper. Does combining these three enhancements beat classical FQL, and is that gain attributable to the combination, not one component, and not some incidental difference in how the two agents were configured? And do CartPole-tuned hyperparameters transfer zero-shot to new environments, or does each environment need its own tuning pass? Both are answered with data, including the honest limitations, not with a single headline number.

**[Slide 3 — Why Fuzzy Q-Learning]**
A trained deep policy is a black box; a fuzzy controller's decisions trace to specific rules and firing strengths. That interpretability matters in safety-critical control, where an operator needs to know why the controller chose an action, not just that it chose well on average. Classical FQL from 1998 uses triangular membership, one-step updates, and fixed exploration -- choices this study tests directly rather than takes on faith, because a twenty-six-year-old design decision deserves a modern re-examination rather than uncritical reuse.

**[Slide 4 — Primary Comparison]**
Across thirty seeds, FQL-GTA solves CartPole -- strictly defined as still above threshold at episode 500 -- in forty percent of seeds, against zero percent for the baseline. Paired Wilcoxon test confirms this with a large effect size, Cohen's d of 2.51, p equals 3.5 times ten to the minus eight. One detail worth stating plainly: the baseline and FQL-GTA use different learning rates by their own frozen configurations, 0.3 versus 0.2, which raised a fair question -- how much of this gap comes from the learning rate rather than the three proposed changes? A dedicated sweep trained the baseline at four different learning rates, same thirty seeds, same episode budget. The best-tuned baseline, at a learning rate of 0.5, still only reaches a final return of 275, against FQL-GTA's 423, with the effect size barely moving, from 2.51 down to 2.50. Tuning the baseline's learning rate does not close the gap. The three proposed changes are what the result is attributable to.

**[Slide 5 — Learning Curves]**
Eighty percent of seeds cross the solve threshold at some point; only forty percent hold it through episode 500. That gap is exactly why "solved" is scored from the final trailing window, not first crossing -- a run that peaks and collapses is not a success, and reporting it as one would overstate how reliable the method actually is.

**[Slide 6 — Full Factorial Ablation]**
All eight combinations of the three enhancements were tested, not just added one at a time. No single enhancement comes close to solving the task alone; even two of three together fall short. A formal factorial effect analysis puts numbers on exactly how the pieces interact. All three main effects are significant -- Gaussian membership is the largest single contributor, ahead of eligibility traces, ahead of adaptive epsilon. Critically, the interaction structure is not uniform: Gaussian membership and eligibility traces amplify each other, a significant positive interaction. Gaussian membership and adaptive epsilon partially offset each other, a significant negative interaction. The eligibility-trace by adaptive-epsilon interaction, and the three-way interaction among all three, are both not significant. So the honest summary is more precise than "the three components synergise" -- two specific pairs interact, one positively and one negatively, and the rest of the improvement comes from the three main effects themselves.

**[Slide 7 — Statistics, Including the Limitation]**
Normality is rejected for both agents by Shapiro-Wilk, so Wilcoxon leads, not a t-test -- and it's significant either way, for what that's worth. Ten-thousand-resample bootstrap confidence intervals are reported alongside the point estimates, not just a bare mean. For the ablation, an omnibus test across all eight conditions is significant, but at only three seeds per condition, none of the twenty-eight Holm-corrected pairwise comparisons hold up individually. That limitation is reported directly, not hidden behind the significant omnibus result -- a reviewer who checks the pairwise table should not be surprised by what they find there.

**[Slide 8 — Hyperparameter Sensitivity]**
The tuned hyperparameters sit at pronounced performance peaks within the tested parameter ranges, not an arbitrary point on a flat surface. Widen the Gaussian membership past its tuned value and performance collapses to near-random, because every rule fires for every state and local specialisation disappears -- the interpretability the whole method is built around depends on that specialisation actually holding.

**[Slide 9 — Cross-Environment Transfer]**
Direct test of the second question: CartPole-tuned hyperparameters applied with no retuning, versus a cheap coordinate search on a held-out validation seed. Zero-shot transfer is weak, especially on Acrobot, where the untuned trailing-fifty return sits around minus three hundred, well short of the practical bar near minus one hundred. Retuning -- at no extra compute cost beyond the sensitivity sweep already performed -- recovers nearly all of the lost performance, reaching about minus one hundred and three on Acrobot and about minus one hundred and twenty-two on MountainCar. The message is not that the method transfers for free; it is that a cheap, already-available retuning step is enough to recover it.

**[Slide 10 — Deep-RL Baselines]**
Benchmarked against a hand-rolled DQN and stable-baselines3's DQN and PPO under a comparable budget. PPO wins on final performance but needs roughly nine thousand one hundred and fifty-five parameters and about thirty-six kilobytes of memory, against FQL-GTA's six hundred and forty-eight parameters and two and a half kilobytes -- roughly fourteen times smaller on both counts. Both DQN variants actually under-perform FQL-GTA within this training budget -- a budget-dependent finding, not a claim that fuzzy Q-learning beats deep RL generally. Given enough additional training, DQN would likely close that gap; the point here is what each method achieves per unit of compute and memory, not an unconditional ranking.

**[Slide 11 — Interpretability, Concretely]**
One real trained-agent decision, fully traced: the state, which of the three hundred twenty-four rules actually fire, the dominant rule's firing strength and Q-values, and the resulting blended decision. Only a handful of rules fire per state -- confirmation that Gaussian membership preserves local specialisation, not uniform activation, and that this interpretability claim is demonstrated on an actual decision, not just argued for in the abstract.

**[Slide 12 — Conclusion]**
Summary: the combination produces a large, statistically robust improvement, driven by significant two-way interaction effects, with no significant three-way interaction, confirmed not to be a learning-rate artifact by a dedicated fairness check, with the ablation's power limitation reported honestly. Zero-shot transfer is weak, but cheap retuning fixes most of it. Against deep RL, this method trades some final performance for an order-of-magnitude smaller model, faster training, and full rule-level interpretability. Two limitations worth naming directly: the rule base grows as the product of per-dimension membership-function counts, so this specific discretisation would need rethinking on a much higher-dimensional state space, and no real industrial deployment has been attempted -- the interpretability and resource-footprint results here make a plausibility case, not a demonstrated production outcome. The contribution is deliberately modest: not new ingredients, but a systematically evaluated, reproducible enhancement to a twenty-six-year-old method, with every claim checked against a specific number rather than asserted.

**[Slide 13 — Thank You]**
Thank you. Code, configuration, results, and the manuscript accompany this presentation.

---
*Tip: speak the numbers in slides 4, 6, 7, 9, and 10 slowly -- they carry the paper's central findings.*
