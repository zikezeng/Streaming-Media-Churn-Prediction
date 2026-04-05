import os
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH  = os.path.join(BASE_DIR, 'Data', 'raw', 'streaming_raw.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'Data', 'processed', 'cleaned.csv')

df = pd.read_csv(INPUT_PATH)

# Drop irrelevant column
df.drop(columns=['Customer_ID'], inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Remove negative values in Monthly_Spend
df = df[df['Monthly_Spend'] >= 0]

# Save cleaned data
df.to_csv(OUTPUT_PATH, index=False)
print(f'✅ Cleaning complete! {len(df)} rows saved.')