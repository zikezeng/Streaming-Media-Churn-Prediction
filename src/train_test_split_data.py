import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'Data', 'processed', 'encoded.csv')

df = pd.read_csv(INPUT_PATH)

# Separate features and target
X = df.drop('Churned', axis=1)
y = df['Churned']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('\nTraining set churn distribution:')
print(y_train.value_counts())

print('\nTesting set churn distribution:')
print(y_test.value_counts())