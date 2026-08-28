# Selected Research Paper — Reference & Reproduction Basis

## Base paper (the work being improved)
**Jouffe, L. (1998).** *Fuzzy inference system learning by reinforcement methods.*
IEEE Transactions on Systems, Man, and Cybernetics — Part C: Applications and
Reviews, 28(3), 338–355. DOI: 10.1109/5326.704563

This is the foundational Fuzzy Q-Learning (FQL) paper. It formalises the
assignment of action qualities to fuzzy rules and their update by
temporal-difference reinforcement, and demonstrates the method on continuous
control problems including pole balancing. The classical agent reproduced in
this submission (triangular membership functions, one-step TD update, fixed
exploration) follows this formulation.

## Reference open implementation used to anchor the reproduction
Masoumzadeh, S. S. — *Fuzzy-Q-Learning* (Python), which provides a clean FQL
implementation with a CartPole pole-balancing example and both triangular and
trapezoidal membership functions. Repository:
https://github.com/seyedsaeidmasoumzadeh/Fuzzy-Q-Learning

> Note on the improvement direction: the classical FQL formulation is enhanced
> here — not merely re-run — by substituting Gaussian membership functions,
> adding Watkins' Q(lambda) eligibility traces, and adopting an adaptive
> exploration schedule (collectively, **FQL-GTA**). This satisfies the
> assignment requirement to take an already-published method and improve it.

## Supporting / comparative references
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning.* Machine Learning, 8, 279–292.
- Shankar, K., Louw, W., & Cohen, K. (2025). *On-policy optimization of ANFIS policies using Proximal Policy Optimization.* arXiv:2507.01039.
- Hein, D., Limmer, S., & Runkler, T. A. (2020). *Interpretable control by reinforcement learning.* arXiv:2007.09964.
- Li, S., et al. (2024). *A fuzzy reinforcement LSTM-based long-term prediction model for fault conditions in nuclear power plants.* arXiv:2411.08370.
- Towers, M., et al. (2024). *Gymnasium: A Standard Interface for Reinforcement Learning Environments.* arXiv:2407.17032.

## How to obtain the base paper PDF for LMS upload
The Jouffe (1998) paper is available through IEEE Xplore. Access it via the
REVA University library subscription (or IEEE Xplore directly) and upload the
downloaded PDF alongside this submission. It is not redistributed here to
respect copyright.
