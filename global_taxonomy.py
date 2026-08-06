import json
from pathlib import Path
from pycocotools.coco import COCO

# =====================================================
# FOOD RECOG 2022
# =====================================================
coco_train = COCO(
    "data_folder/datasets/phase2/foodrecog2022/raw_data/train/annotations.json"
)

fr22_cats = {
    cat["id"]: cat["name"].strip().lower()
    for cat in coco_train.cats.values()
}

print(f"FoodRecog2022 classes : {len(fr22_cats)}")


# =====================================================
# UEC-256
# =====================================================
uec_root = Path("data_folder/datasets/phase2/uec256/UECFOOD256")

category_file = uec_root / "category.txt"

uec_cats = {}

with open(category_file, encoding="utf-8") as f:
    next(f)  # skip header
    for line in f:

        line = line.strip()

        if not line:
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            continue

        cat_id = int(parts[0])
        name = parts[1].strip().lower().replace("_", " ")

        uec_cats[cat_id] = name

print(f"UEC-256 classes       : {len(uec_cats)}")


# =====================================================
# FOODSEG103
# =====================================================
foodseg_root = Path(
    "data_folder/datasets/phase2/foodseg103/FoodSeg103"
)

category_file = foodseg_root / "category_id.txt"

foodseg_cats = {}

with open(category_file, encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            continue

        cat_id = int(parts[0])
        name = parts[1].strip().lower().replace("_", " ")

        foodseg_cats[cat_id] = name

print(f"FoodSeg103 classes    : {len(foodseg_cats)}")




# =====================================================
# COMBINE ALL NAMES
# =====================================================

all_p2_names = set()

all_p2_names.update(fr22_cats.values())
all_p2_names.update(uec_cats.values())
all_p2_names.update(foodseg_cats.values())


all_p2_names = sorted(all_p2_names)

print("\n--------------------------------")
print(f"Unique Phase-2 classes : {len(all_p2_names)}")
print("--------------------------------")

with open(
    "data_folder/merged/phase2_classes.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(all_p2_names))

print("Saved: data_folder/merged/phase2_classes.txt")