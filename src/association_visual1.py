import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

rules_path = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "association_results",
    "cluster0_rules_to_churn.csv"
)

rules = pd.read_csv(rules_path)
top_rules = rules.sort_values(by="lift", ascending=False).head(9).reset_index(drop=True)

def clean_rule_text(x):
    text = str(x)
    text = text.replace("frozenset({", "")
    text = text.replace("})", "")
    text = text.replace("'", "")
    text = text.replace(", ", " + ")
    return text

labels = top_rules["antecedents"].apply(clean_rule_text)
labels = labels.apply(lambda s: "\n".join(textwrap.wrap(s, width=26)))

colors = []
for i in range(len(top_rules)):
    if i < 3:
        colors.append("#d62728")
    elif i < 6:
        colors.append("#ff7f0e")
    else:
        colors.append("#2ca02c")

fig = plt.figure(figsize=(16, 8))
gs = GridSpec(1, 2, width_ratios=[2.6, 1.6], wspace=0.02, figure=fig)


ax_text = fig.add_subplot(gs[0, 0])
ax_text.set_xlim(0, 1)
ax_text.set_ylim(-0.5, len(top_rules) - 0.5)
ax_text.invert_yaxis()
ax_text.axis("off")

for i, label in enumerate(labels):
    ax_text.text(
        0.98, i, label,
        ha="right", va="center",
        fontsize=10
    )

# 右边：柱状图区
ax_bar = fig.add_subplot(gs[0, 1])
ax_bar.barh(range(len(top_rules)), top_rules["lift"], color=colors)
ax_bar.set_ylim(-0.5, len(top_rules) - 0.5)
ax_bar.invert_yaxis()
ax_bar.set_xlabel("Lift")
ax_bar.set_title("Top Churn-Related Feature Combinations")
ax_bar.set_yticks([])
ax_bar.spines["left"].set_visible(False)



plot_path = os.path.join(OUTPUT_DIR, "top_churn_rules_colored.png")
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", os.path.abspath(plot_path))