import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'Data', 'processed', 'encoded.csv')

df = pd.read_csv(INPUT_PATH)

# Separate features and target
X = df.drop('Churned', axis=1)
y = df['Churned']

print('X shape:', X.shape)
print('y shape:', y.shape)
print(y.value_counts())