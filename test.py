from pathlib import Path

root = Path("data_folder/merged/phase2/images")

count = 0

for f in root.rglob("*.npy"):
    f.unlink()
    count += 1

print(f"Deleted {count} cache files.")