from pycocotools.coco import COCO

coco_train = COCO('data_folder/datasets/phase2/foodrecog2022/raw_data/train/annotations.json')
coco_val   = COCO('data_folder/datasets/phase2/foodrecog2022/raw_data/val/annotations.json')

# Preview annotation structure:
sample_ann = coco_train.loadAnns([1])[0]
print(sample_ann)
# {'id': 1, 'image_id': 123, 'category_id': 42,
#  'bbox': [x, y, w, h], 'segmentation': [...]}
import json
import shutil
from pathlib import Path

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

def convert_coco_to_yolo(coco, img_dir, out_img_dir, out_lbl_dir, prefix):
  for img_id, img_info in coco.imgs.items():
    W, H = img_info['width'], img_info['height']
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns    = coco.loadAnns(ann_ids)
    if not anns: continue   # skip unannotated images

    src = Path(img_dir) / img_info['file_name']
    name = f'{prefix}_{img_info["file_name"]}'
    dst_img = Path(out_img_dir) / name
    dst_lbl = Path(out_lbl_dir) / (Path(name).stem + '.txt')

    if not src.exists(): continue
    shutil.copy(src, dst_img)

    with open(dst_lbl, 'w') as f:
      for ann in anns:
        x, y, w, h = ann['bbox']
        if w*h == 0: continue   # skip zero-area boxes
        cx = (x + w/2) / W
        cy = (y + h/2) / H
        nw, nh = w/W, h/H
        cat_name = coco.cats[ann['category_id']]['name']
        normalized_name = normalize(cat_name)
        canonical_name = dedup_map.get(normalized_name, normalized_name)
        global_id = global_map.get(canonical_name)
        if global_id is None: continue
        f.write(f'{global_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n')

convert_coco_to_yolo(
  coco_train,
  'data_folder/datasets/phase2/foodrecog2022/raw_data/train/images',
  'data_folder/merged/phase2/images/train',
  'data_folder/merged/phase2/labels/train',
  'fr22'
)