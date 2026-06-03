# scripts/check_labels.py
import sys
import pandas as pd
from collections import Counter

if len(sys.argv) < 2:
    print("Usage: python scripts/check_labels.py path/to/train.csv")
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)
print("HEAD:\n", df.head(5))
print("\nColumns:", df.columns.tolist())

if 'label' not in df.columns:
    print("❌ no 'label' column found. Make sure CSV has 'image_path,mask_path,label'")
    sys.exit(1)

print("\nLabel dtype:", df['label'].dtype)
print("\nLabel counts:")
print(df['label'].value_counts())
print("\nCounter:", Counter(df['label'].tolist()))

