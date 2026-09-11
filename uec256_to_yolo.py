import shutil

import cv2, os
import json
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

uec_cats = {}
with open("data_folder/datasets/phase2/uec256/UECFOOD256/category.txt", encoding="utf-8") as f:
    next(f)
    for line in f:
        line = line.strip()
        if not line: continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            uec_cats[str(parts[0])] = parts[1].strip().lower().replace("_", " ")

def parse_uec256(uec_root):
  uec_root = Path(uec_root)
  # Walk each category folder (named 1, 2, ..., 256)
  for cat_folder in sorted(uec_root.iterdir()):
    if not cat_folder.is_dir(): continue
    cat_id = cat_folder.name   # numeric string '1'..'256'
    bb_file = cat_folder / 'bb_info.txt'
    if not bb_file.exists(): continue
    with open(bb_file) as f:
      lines = f.readlines()[1:]  # skip header
    for line in lines:
      parts = line.strip().split()
      if len(parts) < 5: continue
      img_id, x1, y1, x2, y2 = parts[:5]
      yield cat_id, img_id, int(x1),int(y1),int(x2),int(y2)



for cat_id, img_id, x1,y1,x2,y2 in parse_uec256('data_folder/datasets/phase2/uec256/UECFOOD256'):
  img_path = Path(f'data_folder/datasets/phase2/uec256/UECFOOD256/{cat_id}/{img_id}.jpg')
  if not img_path.exists(): continue
  H, W = cv2.imread(str(img_path)).shape[:2]
  cx = ((x1+x2)/2) / W
  cy = ((y1+y2)/2) / H
  bw = (x2-x1) / W
  bh = (y2-y1) / H
  # Clamp to [0,1] in case of annotation errors
  cx,cy,bw,bh = [max(0,min(1,v)) for v in [cx,cy,bw,bh]]
  
  cat_name = uec_cats.get(str(cat_id))
  if not cat_name: continue
  
  normalized_name = normalize(cat_name)
  canonical_name = dedup_map.get(normalized_name, normalized_name)
  global_id = global_map.get(canonical_name)
  
  if global_id is None: continue

  dst_lbl = Path(f'data_folder/merged/phase2/labels/train/uec_{img_id}.txt')
  with open(dst_lbl, 'a') as f:
    f.write(f'{global_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
  shutil.copy(img_path, f'data_folder/merged/phase2/images/train/uec_{img_id}.jpg')