# src/split.py
import os
import random
import json
import shutil

# Source directory
source_dir = "/home/minhchau/anaconda3/envs/curvas/curvas/Data"
# Output directory
output_dir = "/home/minhchau/anaconda3/envs/curvas/curvas/Split_Data"

# Get all case folders
cases = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])

# Shuffle with fixed seed for reproducibility
random.seed(42)
random.shuffle(cases)

# Split 80% train, 20% val
n = len(cases)
n_train = int(n * 0.8)
train_cases = cases[:n_train]
val_cases = cases[n_train:]

# Create output folders
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

print(f"Total cases: {n}")
print(f"Train: {len(train_cases)} cases (80%)")
print(f"Val: {len(val_cases)} cases (20%)")
print(f"\nCreating folders at: {output_dir}")

# Copy train cases
print(f"\nCopying {len(train_cases)} cases to train folder...")
for case in train_cases:
    src = os.path.join(source_dir, case)
    dst = os.path.join(train_dir, case)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
print(f"✅ Copied {len(train_cases)} train cases")

# Copy val cases
print(f"\nCopying {len(val_cases)} cases to val folder...")
for case in val_cases:
    src = os.path.join(source_dir, case)
    dst = os.path.join(val_dir, case)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
print(f"✅ Copied {len(val_cases)} val cases")

# Save split.json
out = {"train": train_cases, "val": val_cases}
split_json_path = os.path.join(output_dir, "split.json")
with open(split_json_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n✅ Saved split.json to {split_json_path}")
print(f"\n📁 Output structure:")
print(f"   {output_dir}/")
print(f"   ├── train/  ({len(train_cases)} cases)")
print(f"   ├── val/    ({len(val_cases)} cases)")
print(f"   └── split.json")
