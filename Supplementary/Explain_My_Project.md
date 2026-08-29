# Explain My Project — FQL-GTA (plain-language / viva prep)

Use this as a spoken answer, not a document to read aloud verbatim — say it in your own words.

## What is it?

It's an improved version of a 1998 reinforcement-learning method called Fuzzy Q-Learning. Instead
of a black-box neural network deciding actions, it uses human-readable if-then fuzzy rules -- so
you can actually see why the agent chose an action, not just that it worked.

## What did I do?

I took the original 1998 method and upgraded three specific parts of it:
1. **Gaussian membership functions** instead of triangular ones -- smoother, more accurate rule
   boundaries.
2. **Eligibility traces** -- when the agent gets a reward, credit spreads back across the recent
   states that led there, not just the very last one.
3. **Adaptive exploration** -- the agent explores a lot early on and settles down over time,
   instead of exploring at a fixed rate forever.

I called the combination **FQL-GTA** (Gaussian, Trace, Adaptive) and tested it on CartPole -- the
classic "balance a pole on a moving cart" benchmark.

## How does it work?

The agent splits the cart's state (position, speed, pole angle, angular speed) into fuzzy rules.
Each rule "fires" with some strength depending on the current state, and the agent's decision is a
blend of all the rules that are firing, weighted by how strongly they fire. Learning just means
adjusting how much each rule's action is worth, based on trial and error.

## What did I find?

Across 30 independent runs, FQL-GTA solved the task 40% of the time; the classical version solved
it 0% of the time -- a large, statistically solid improvement (Cohen's d = 2.51, p < 0.0001). I
didn't take that at face value, though:
- A full factorial test of all 8 combinations of the three upgrades shows all three matter, two of
  them boost each other, and one partially cancels another -- not a blanket "everything helps
  equally" story.
- I specifically checked whether the improvement was just because the two agents used different
  learning-rate settings by their frozen configs -- it isn't; even the best-tuned classical version
  (found by sweeping four learning rates) still loses badly.
- Tuned settings mostly don't transfer zero-shot to other environments, but a cheap retuning step
  fixes that.
- Against deep RL (PPO), PPO performs better but needs about 14 times more parameters and memory.

## Why does it matter?

In safety-critical control -- robotics, industrial systems -- you often need to know *why* a
controller did something, not just trust that it usually works. This method keeps that
explainability while still being statistically competitive.

## If they push further

- **"Why not just use deep RL?"** -- Deep RL wins on raw performance here, but at ~14x the model
  size and with no rule-level explanation for any single decision. The trade-off is the point.
- **"Isn't the ablation underpowered?"** -- Yes, and I say so directly: only 3 seeds per condition
  means the pairwise comparisons aren't individually significant even though the overall pattern
  is. Reported as a real limitation, not hidden.
- **"How do you know it's not overfit to CartPole?"** -- I tested transfer to two other
  environments (MountainCar, Acrobot) and reported honestly that zero-shot transfer is weak.
