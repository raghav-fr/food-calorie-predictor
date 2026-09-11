import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# Paths
# ----------------------------
ROOT = Path("data_folder/datasets/phase2/foodseg103/FoodSeg103")

IMG_ROOT = ROOT / "Images" / "img_dir"
MASK_ROOT = ROOT / "Images" / "ann_dir"

OUT_ROOT = Path("data_folder/merged/phase2")

TRAIN_IMG_OUT = OUT_ROOT / "images" / "train"
TRAIN_LBL_OUT = OUT_ROOT / "labels" / "train"

VAL_IMG_OUT = OUT_ROOT / "images" / "val"
VAL_LBL_OUT = OUT_ROOT / "labels" / "val"
with open("data_folder/logs/dedup_map.json", encoding="utf-8") as f:
    dedup_map = json.load(f)["mapping"]

with open("data_folder/merged/global_class_map.json", encoding="utf-8") as f:
    global_map = json.load(f)["map"]

def normalize(name):
    return (
        name.lower()
            .replace("_", " ")
            .replace("-", " ")
            .replace("&", "and")
            .strip()
    )

foodseg_cats = {}
with open(ROOT / "category_id.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            foodseg_cats[int(parts[0])] = parts[1].strip().lower().replace("_", " ")

for p in [
    TRAIN_IMG_OUT,
    TRAIN_LBL_OUT,
    VAL_IMG_OUT,
    VAL_LBL_OUT
]:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Convert one mask to YOLO boxes
# ----------------------------
def mask_to_bboxes(mask_path, img_w, img_h):

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return []

    boxes = []

    classes = np.unique(mask)

    for class_id in classes:

        # background
        if class_id == 0:
            continue

        binary = (mask == class_id).astype(np.uint8)

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)

        # component 0 is background
        for i in range(1, n_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # remove tiny noisy regions
            if area < 100:
                continue

            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h

            bw = w / img_w
            bh = h / img_h

            boxes.append(
                (
                    int(class_id),
                    cx,
                    cy,
                    bw,
                    bh,
                )
            )

    return boxes


# ----------------------------
# Process train/test split
# ----------------------------
def process_split(split):

    if split == "train":
        out_img = TRAIN_IMG_OUT
        out_lbl = TRAIN_LBL_OUT
    else:
        out_img = VAL_IMG_OUT
        out_lbl = VAL_LBL_OUT

    img_dir = IMG_ROOT / split
    mask_dir = MASK_ROOT / split

    images = sorted(img_dir.glob("*.jpg"))

    print(f"\nProcessing {split}: {len(images)} images")

    total_boxes = 0

    for img_path in tqdm(images):

        mask_path = mask_dir / (img_path.stem + ".png")

        if not mask_path.exists():
            continue

        image = cv2.imread(str(img_path))

        if image is None:
            continue

        H, W = image.shape[:2]

        boxes = mask_to_bboxes(mask_path, W, H)

        if len(boxes) == 0:
            continue

        new_name = f"seg_{img_path.name}"

        shutil.copy(
            img_path,
            out_img / new_name
        )

        label_path = out_lbl / f"seg_{img_path.stem}.txt"

        with open(label_path, "w") as f:

            for class_id, cx, cy, bw, bh in boxes:

                cat_name = foodseg_cats.get(int(class_id))
                if not cat_name: continue
                
                normalized_name = normalize(cat_name)
                canonical_name = dedup_map.get(normalized_name, normalized_name)
                global_id = global_map.get(canonical_name)
                
                if global_id is None: continue

                f.write(
                    f"{global_id} "
                    f"{cx:.6f} "
                    f"{cy:.6f} "
                    f"{bw:.6f} "
                    f"{bh:.6f}\n"
                )

                total_boxes += 1

    print(f"Finished {split}")
    print(f"Total boxes: {total_boxes}")


# ----------------------------
# Run
# ----------------------------
process_split("train")
process_split("test")

print("\nFoodSeg103 conversion completed.")