
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'Data', 'processed', 'clustered.csv')
df = pd.read_csv(INPUT_PATH)

cluster_labels = {
    0: 'At-Risk Mid-Tenure',
    1: 'New Loyal',
    2: 'Established Loyal'
}

# Discount elasticity per cluster

elasticity_results = []
for cid in sorted(df['Cluster'].unique()):
    sub = df[df['Cluster'] == cid]
    X = sub[['Discount_Offered']].values
    y = sub['Monthly_Spend'].values

    lr = LinearRegression().fit(X, y)
    slope = lr.coef_[0]
    r_squared = r2_score(y, lr.predict(X))
    r_corr, p_value = stats.pearsonr(sub['Discount_Offered'], sub['Monthly_Spend'])

    elasticity_results.append({
        'Cluster': cid,
        'Label': cluster_labels[cid],
        'Size': len(sub),
        'Avg_Discount(%)': round(sub['Discount_Offered'].mean(), 2),
        'Avg_Spend($)': round(sub['Monthly_Spend'].mean(), 2),
        'Elasticity($/1%)': round(slope, 4),
        'R_squared': round(r_squared, 4),
        'P_value': f'{p_value:.2e}',
        'Churn_Rate': round(sub['Churned'].mean(), 4)
    })

elasticity_df = pd.DataFrame(elasticity_results)
print(elasticity_df.to_string(index=False))
print()
print('Note: Low R² is expected — discount alone explains little within-cluster')
print('      variation. The elasticity coefficient (slope) and its p-value are')
print('      the relevant outputs for comparing segment-level discount response.')
print()

# Net Revenue analysis

print('Net_Revenue = Monthly_Spend * (1 - Discount/100)')
df['Net_Revenue'] = df['Monthly_Spend'] * (1 - df['Discount_Offered'] / 100)
df['Discount_Bucket'] = pd.cut(
    df['Discount_Offered'],
    bins=[0, 9, 13, 17, 25],
    labels=['Low(5-9%)', 'Mid-Low(9-13%)', 'Mid-High(13-17%)', 'High(17-20%)']
)

net_revenue_table = df.pivot_table(
    values='Net_Revenue', index='Cluster',
    columns='Discount_Bucket', aggfunc='mean', observed=True
).round(2)
net_revenue_table.index = [f'C{c}: {cluster_labels[c]}' for c in net_revenue_table.index]
print('Average Net Revenue per Customer ($):')
print(net_revenue_table)
print()

print('Net Revenue change from Low to High discount:')
for cid in sorted(df['Cluster'].unique()):
    sub = df[df['Cluster'] == cid]
    low = sub[sub['Discount_Bucket'] == 'Low(5-9%)']['Net_Revenue'].mean()
    high = sub[sub['Discount_Bucket'] == 'High(17-20%)']['Net_Revenue'].mean()
    change_pct = (high - low) / low * 100
    print(f'  Cluster {cid} ({cluster_labels[cid]}): '
          f'${low:.2f} -> ${high:.2f}  ({change_pct:+.1f}%)')
print()

# Visualization

results_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(results_dir, exist_ok=True)

colors = {0: '#E63946', 1: '#F4A261', 2: '#2A9D8F'}
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: Gross spend
ax = axes[0]
for cid in sorted(df['Cluster'].unique()):
    sub = df[df['Cluster'] == cid].groupby('Discount_Bucket', observed=True)['Monthly_Spend'].mean()
    ax.plot(sub.index.astype(str), sub.values, marker='o', linewidth=2.5,
            markersize=10, label=f'C{cid}: {cluster_labels[cid]}', color=colors[cid])
ax.set_xlabel('Discount Level', fontsize=11)
ax.set_ylabel('Average Monthly Spend ($)', fontsize=11)
ax.set_title('Gross Revenue: Spend Increases with Discount', fontsize=12, pad=10)
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)

# Right: Net revenue
ax = axes[1]
for cid in sorted(df['Cluster'].unique()):
    sub = df[df['Cluster'] == cid].groupby('Discount_Bucket', observed=True)['Net_Revenue'].mean()
    ax.plot(sub.index.astype(str), sub.values, marker='o', linewidth=2.5,
            markersize=10, label=f'C{cid}: {cluster_labels[cid]}', color=colors[cid])
ax.set_xlabel('Discount Level', fontsize=11)
ax.set_ylabel('Net Revenue per Customer ($)', fontsize=11)
ax.set_title('Net Revenue: Discount Erodes Profit', fontsize=12, pad=10)
ax.legend(loc='center right', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(results_dir, 'discount_elasticity_analysis.png')
plt.savefig(fig_path, dpi=180, bbox_inches='tight')
plt.close()
print(f'Visualization saved to: {fig_path}')