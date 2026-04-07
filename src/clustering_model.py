import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'Data', 'processed', 'cleaned.csv')

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

# Show churn rate by cluster
print('Churn rate by cluster:')
print(df.groupby('Cluster')['Churned'].mean())

print('\nCluster sizes:')
print(df['Cluster'].value_counts())

print('\nCluster summary:')
print(df.groupby('Cluster')[[
    'Age',
    'Subscription_Length',
    'Support_Tickets_Raised',
    'Satisfaction_Score',
    'Discount_Offered',
    'Last_Activity',
    'Monthly_Spend',
    'Churned'
]].mean())