import os

root = "data_folder/datasets/phase2/foodseg103"

for path, dirs, files in os.walk(root):
    print(path)
    if len(files):
        print("  Files:", files[:5])