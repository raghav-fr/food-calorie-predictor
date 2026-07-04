
from pathlib import Path

import yaml, os

with open('data_folder/merged/phase1_classes.txt', encoding='utf-8') as f:
  class_list = [l.strip() for l in f]

yaml_content = {
  'path':  os.path.abspath('data_folder/merged/phase1'),
  'train': 'images/train',
  'val':   'images/val',
  'nc':    len(class_list),
  'names': class_list
}

with open('data_phase1.yaml', 'w', encoding='utf-8') as f:
  yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

print(f'data_phase1.yaml written — {len(class_list)} classes')

for subset in ['train','val']:
  imgs = {p.stem for p in
          Path(f'data_folder/merged/phase1/images/{subset}').glob('*')}
  lbls = {p.stem for p in
          Path(f'data_folder/merged/phase1/labels/{subset}').glob('*.txt')}
  missing_lbl  = imgs - lbls
  missing_img  = lbls - imgs
  print(f'{subset}: {len(imgs)} images, {len(lbls)} labels')
  print(f'  Missing labels: {len(missing_lbl)}')
  print(f'  Orphan labels:  {len(missing_img)}')