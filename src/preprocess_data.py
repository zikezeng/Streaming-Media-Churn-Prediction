import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'Data', 'processed', 'cleaned.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'Data', 'processed', 'encoded.csv')

df = pd.read_csv(INPUT_PATH)

# Encode categorical variables
df_encoded = pd.get_dummies(
    df,
    columns=['Gender', 'Region', 'Payment_Method'],
    drop_first=True
)

# Save processed data
df_encoded.to_csv(OUTPUT_PATH, index=False)

print(f'Preprocessing complete: {len(df_encoded)} rows saved.')
