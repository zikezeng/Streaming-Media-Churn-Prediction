import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rules_path = os.path.join(
    BASE_DIR,
    'Data',
    'processed',
    'association_results_no_satisfaction',
    'cluster0_no_satisfaction_rules_to_churn.csv'
)

OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
rules = pd.read_csv(rules_path)
# =========================
top_rules = rules.sort_values(by='lift', ascending=False).head(9).reset_index(drop=True)

# =========================
def clean_text(x):
    return str(x).replace("frozenset({", "").replace("})", "")

labels = top_rules['antecedents'].apply(clean_text)
# =========================
colors = []
for i in range(len(top_rules)):
    if i < 3:
        colors.append('#d62728')  # red-high risk
    elif i < 6:
        colors.append('#ff7f0e')  # orange-middle risk
    else:
        colors.append('#2ca02c')  # green-low risk

# =========================
plt.figure(figsize=(8, 5))

plt.barh(range(len(top_rules)), top_rules['lift'], color=colors)

plt.yticks(range(len(top_rules)), labels)
plt.xlabel('Lift')
plt.title('Churn Drivers Without Satisfaction')

plt.gca().invert_yaxis()
plt.tight_layout()

# =========================
# Figure 5
# =========================
plt.figtext(
    0.5, -0.08,
    'Figure 5. Churn Drivers Without Satisfaction',
    ha='center',
    fontsize=10
)

# =========================
# save
# =========================
plot_path = os.path.join(OUTPUT_DIR, 'churn_without_satisfaction_colored.png')

plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print("Saved to:", os.path.abspath(plot_path))