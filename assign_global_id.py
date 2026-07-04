import json

with open('data_folder/merged/phase1_classes.txt', encoding='utf-8') as f:
  class_list = [l.strip() for l in f]

class_to_id = {name: i for i, name in enumerate(class_list)}
id_to_class = {i: name for name, i in class_to_id.items()}

with open('data_folder/merged/phase1_class_map.json','w',encoding='utf-8') as f:
  json.dump({'class_to_id': class_to_id, 'id_to_class': id_to_class}, f,
            ensure_ascii=False, indent=2)

print(f'Total Phase 1 classes: {len(class_list)}')