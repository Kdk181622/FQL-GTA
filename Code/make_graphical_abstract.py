"""Renders the graphical abstract / pipeline diagram for FQL-GTA: the full
decision cycle from raw state to selected action, using plain matplotlib
shapes so it needs no external diagramming tool or dependency."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "Figures"
FIG_DIR.mkdir(exist_ok=True)

STAGES = [
    ("State", "Continuous environment observation\n(e.g. cart position, pole angle)"),
    ("Fuzzification", "Gaussian membership functions map\neach state dimension to firing degrees"),
    ("Rule Base", "Product t-norm combines per-dimension\ndegrees into per-rule firing strengths"),
    ("Q-Learning", "Firing-strength-weighted sum of\nper-rule parameters approximates Q(s,a)"),
    ("Eligibility Traces", "TD error propagated across recently\nactive rules via Watkins' Q(lambda)"),
    ("Adaptive Epsilon", "Exploration rate anneals from a high\nstarting value toward a small floor"),
    ("Optimal Action", "Epsilon-greedy selection over the\nupdated action-value estimates"),
]

COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2", "#937860", "#64B5CD"]


def build():
    fig, ax = plt.subplots(figsize=(4.6, 11.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(STAGES) * 2 + 1)
    ax.axis("off")

    box_w, box_h = 8.4, 1.5
    x0 = (10 - box_w) / 2
    for i, ((title, detail), color) in enumerate(zip(STAGES, COLORS)):
        y = len(STAGES) * 2 - i * 2
        box = FancyBboxPatch((x0, y - box_h / 2), box_w, box_h,
                              boxstyle="round,pad=0.08,rounding_size=0.15",
                              linewidth=1.2, edgecolor="black",
                              facecolor=color, alpha=0.85)
        ax.add_patch(box)
        ax.text(5, y + 0.28, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        ax.text(5, y - 0.32, detail, ha="center", va="center",
                fontsize=8.3, color="white")

        if i < len(STAGES) - 1:
            arrow = FancyArrowPatch((5, y - box_h / 2 - 0.05), (5, y - 2 + box_h / 2 + 0.05),
                                     arrowstyle="-|>", mutation_scale=18,
                                     linewidth=1.4, color="black")
            ax.add_patch(arrow)

    # Loop-back arrow: the selected action closes the decision cycle back to
    # a new state observation, which the linear stack alone doesn't show.
    ax.annotate("", xy=(x0 - 0.3, len(STAGES) * 2), xytext=(x0 - 0.3, 1),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.2, color="gray",
                                 connectionstyle="arc3,rad=-0.15"))
    ax.text(x0 - 0.9, len(STAGES), "environment\nstep", rotation=90, ha="center",
            va="center", fontsize=8, color="gray")

    ax.set_title("FQL-GTA: State-to-Action Decision Cycle", fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_graphical_abstract.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved", FIG_DIR / "fig10_graphical_abstract.png")


if __name__ == "__main__":
    build()
