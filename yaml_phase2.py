import json

import yaml, os

with open('data_folder/merged/global_class_map.json', encoding='utf-8') as f:
  gmap = json.load(f)

yaml_content = {
  'path':  os.path.abspath('data_folder/merged/phase2'),
  'train': 'images/train',
  'val':   'images/val',
  'nc':    len(gmap['names']),
  'names': gmap['names']
}
with open('data_phase2.yaml','w',encoding='utf-8') as f:
  yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

print(f'data_phase2.yaml: {len(gmap["names"])} classes')