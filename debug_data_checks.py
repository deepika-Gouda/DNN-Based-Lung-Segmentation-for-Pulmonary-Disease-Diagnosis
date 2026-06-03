# debug_data_checks.py
import argparse
import pandas as pd
import os

def print_csv_stats(path, name):
    df = pd.read_csv(path)
    print(f"\n--- {name} ---")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    if all(c in df.columns for c in ["image_path","mask_path","label"]):
        print("Label value counts:\n", df["label"].value_counts())
        # First 5 rows
        print("\nFirst 5 rows:")
        print(df.head())
    else:
        print("CSV missing required columns (image_path, mask_path, label).")

def check_overlap(train_csv, val_csv):
    df1 = pd.read_csv(train_csv)
    df2 = pd.read_csv(val_csv)
    set1 = set(df1["image_path"].apply(lambda p: os.path.basename(p)))
    set2 = set(df2["image_path"].apply(lambda p: os.path.basename(p)))
    inter = set1.intersection(set2)
    print(f"\nImage basenames overlap between train & val: {len(inter)}")
    if len(inter) > 0:
        print("Some overlapping files (showing up to 10):", list(inter)[:10])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    args = parser.parse_args()
    print_csv_stats(args.train_csv, "TRAIN CSV")
    print_csv_stats(args.val_csv, "VAL CSV")
    check_overlap(args.train_csv, args.val_csv)
