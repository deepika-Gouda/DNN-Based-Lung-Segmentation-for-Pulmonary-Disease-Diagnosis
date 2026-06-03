import pandas as pd

# Mapping class names to integer labels
class_map = {
    "COVID": 0,
    "Normal": 1,
    "Lung_Opacity": 2,
    "Viral Pneumonia": 3
}

def add_labels(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    # Extract class name from the file path
    df['label'] = df['image_path'].apply(lambda x: class_map[x.split('/')[1]])
    
    df.to_csv(output_path, index=False)
    print(f"✅ Saved updated CSV with labels: {output_path}")

# Update train.csv and val.csv
add_labels("train.csv", "train_labeled.csv")
add_labels("val.csv", "val_labeled.csv")
