import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'Data', 'processed', 'cleaned.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output', 'figures')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load cleaned data
df = pd.read_csv(INPUT_PATH)

# Select numerical features for clustering
cluster_features = df[[
    'Age',
    'Subscription_Length',
    'Support_Tickets_Raised',
    'Satisfaction_Score',
    'Discount_Offered',
    'Last_Activity',
    'Monthly_Spend'
]]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

# Build KMeans model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Create cluster summary
cluster_summary = df.groupby('Cluster').agg(
    Size=('Churned', 'size'),
    Churn_Rate=('Churned', 'mean'),
    Age=('Age', 'mean'),
    Subscription_Length=('Subscription_Length', 'mean'),
    Support_Tickets_Raised=('Support_Tickets_Raised', 'mean'),
    Satisfaction_Score=('Satisfaction_Score', 'mean'),
    Discount_Offered=('Discount_Offered', 'mean'),
    Last_Activity=('Last_Activity', 'mean'),
    Monthly_Spend=('Monthly_Spend', 'mean')
).reset_index()

# Convert churn rate to percentage
cluster_summary['Churn_Rate_Percent'] = cluster_summary['Churn_Rate'] * 100

# Rename clusters for clearer interpretation
cluster_names = {
    0: 'At-Risk Mid-Tenure',
    1: 'New Loyal',
    2: 'Established Loyal'
}

cluster_summary['Cluster_Name'] = cluster_summary['Cluster'].map(cluster_names)

# Save cluster summary table
summary_path = os.path.join(OUTPUT_DIR, 'cluster_summary.csv')
cluster_summary.to_csv(summary_path, index=False)

print('Cluster summary:')
print(cluster_summary)
print('\nCluster summary saved to:', summary_path)

# Figure colors
colors = ['#E63946', '#F4A261', '#2A9D8F']

# Figure 1: Customer Cluster Profile by Satisfaction and Monthly Spend
plt.figure(figsize=(9, 6))

plt.scatter(
    cluster_summary['Monthly_Spend'],
    cluster_summary['Satisfaction_Score'],
    s=cluster_summary['Size'] / 3,
    c=colors,
    alpha=0.75
)

for i in range(len(cluster_summary)):
    plt.text(
        cluster_summary.loc[i, 'Monthly_Spend'] + 0.5,
        cluster_summary.loc[i, 'Satisfaction_Score'] + 0.05,
        cluster_summary.loc[i, 'Cluster_Name'],
        fontsize=10
    )

plt.title('Figure 1. Customer Cluster Profile by Satisfaction and Monthly Spend')
plt.xlabel('Average Monthly Spend (USD)')
plt.ylabel('Average Satisfaction Score (1-10)')
plt.grid(True, alpha=0.3)
plt.tight_layout()

figure1_path = os.path.join(OUTPUT_DIR, 'figure1_cluster_profile.png')
plt.savefig(figure1_path, dpi=300)
plt.show()

print('Figure 1 saved to:', figure1_path)

# Figure 2: Churn Rate by Customer Cluster
plt.figure(figsize=(9, 6))

bars = plt.bar(
    cluster_summary['Cluster_Name'],
    cluster_summary['Churn_Rate_Percent'],
    color=colors
)

plt.title('Figure 2. Churn Rate by Customer Cluster')
plt.xlabel('Customer Cluster')
plt.ylabel('Churn Rate (%)')
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 2,
        str(round(height, 1)) + '%',
        ha='center',
        va='bottom'
    )

plt.xticks(rotation=15)
plt.tight_layout()

figure2_path = os.path.join(OUTPUT_DIR, 'figure2_churn_rate_by_cluster.png')
plt.savefig(figure2_path, dpi=300)
plt.show()

print('Figure 2 saved to:', figure2_path)

# Figure 3: Subscription Length by Customer Cluster
plt.figure(figsize=(10, 6))

data_by_cluster = [
    df[df['Cluster'] == 0]['Subscription_Length'],
    df[df['Cluster'] == 1]['Subscription_Length'],
    df[df['Cluster'] == 2]['Subscription_Length']
]

box = plt.boxplot(
    data_by_cluster,
    labels=[
        'Cluster 0:\nAt-Risk Mid-Tenure',
        'Cluster 1:\nNew Loyal',
        'Cluster 2:\nEstablished Loyal'
    ],
    patch_artist=True,
    showfliers=True
)

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

means = [
    cluster_summary.loc[cluster_summary['Cluster'] == 0, 'Subscription_Length'].values[0],
    cluster_summary.loc[cluster_summary['Cluster'] == 1, 'Subscription_Length'].values[0],
    cluster_summary.loc[cluster_summary['Cluster'] == 2, 'Subscription_Length'].values[0]
]

for i in range(3):
    plt.scatter(i + 1, means[i], marker='D', s=80, color='white', edgecolor='black', zorder=3)
    plt.text(
        i + 1.15,
        means[i],
        'mean = ' + str(round(means[i], 1)) + ' mo',
        fontsize=10,
        va='center'
    )

plt.title('Figure 3. Subscription Length by Customer Cluster')
plt.ylabel('Subscription Length (months)')
plt.xlabel('Customer Cluster')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

figure3_path = os.path.join(OUTPUT_DIR, 'figure3_subscription_length_by_cluster.png')
plt.savefig(figure3_path, dpi=300)
plt.show()

print('Figure 3 saved to:', figure3_path)