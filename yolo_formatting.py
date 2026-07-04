from pathlib import Path
import random
import shutil


def pseudo_label_line(class_id: int) -> str:
  return f'{class_id} 0.5 0.5 1.0 1.0\n'


def show_progress(current: int, total: int, label: str) -> None:
  if total <= 0:
    return
  percent = current / total
  filled = int(30 * percent)
  bar = '█' * filled + '░' * (30 - filled)
  print(f'\r{label}: |{bar}| {current}/{total} ({percent:6.2%})', end='', flush=True)


with open('data_folder/merged/phase1_classes.txt', encoding='utf-8') as f:
  class_list = [l.strip() for l in f]

class_to_id = {name: i for i, name in enumerate(class_list)}
id_to_class = {i: name for name, i in class_to_id.items()}

random.seed(42)
root = Path('data_folder/datasets/phase1/food101/images')
class_folders = [p for p in root.iterdir() if p.is_dir()]
food101_total = sum(len(list(class_folder.glob('*.jpg'))) for class_folder in class_folders)
processed = 0

for class_folder in class_folders:
  class_name = class_folder.name.lower().replace('_',' ')
  class_id   = class_to_id.get(class_name)
  if class_id is None: continue

  images = list(class_folder.glob('*.jpg'))
  random.shuffle(images)
  split = int(len(images)*0.9)
  splits = {'train': images[:split], 'val': images[split:]}

  for subset, imgs in splits.items():
    for src in imgs:
      dst_img = Path(f'data_folder/merged/phase1/images/{subset}/{src.name}')
      dst_lbl = Path(f'data_folder/merged/phase1/labels/{subset}/{src.stem}.txt')
      shutil.copy(src, dst_img)
      dst_lbl.write_text(f'{class_id} 0.5 0.5 1.0 1.0\n')
      processed += 1
      show_progress(processed, food101_total, 'Food-101')

print('\nFood-101 done.')

def convert_classification_dataset(src_root, prefix, subset_ratio=0.9):
  src_root = Path(src_root)
  class_folders = [p for p in src_root.iterdir() if p.is_dir()]
  total_images = 0
  for class_folder in class_folders:
    images = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
    total_images += len(images)

  processed = 0
  for class_folder in class_folders:
    if not class_folder.is_dir(): continue
    class_name = class_folder.name.strip().lower().replace('_',' ')
    class_id   = class_to_id.get(class_name)
    if class_id is None:
      print(f'[WARN] unknown class: {class_name}')
      continue
    images = list(class_folder.glob('*.jpg')) + \
             list(class_folder.glob('*.png'))
    random.shuffle(images)
    split  = int(len(images)*subset_ratio)
    for subset, imgs in [('train',images[:split]),('val',images[split:])]:
      for src in imgs:
        name = f'{prefix}_{src.name}'
        shutil.copy(src, f'data_folder/merged/phase1/images/{subset}/{name}')
        Path(f'data_folder/merged/phase1/labels/{subset}/{Path(name).stem}.txt'
             ).write_text(f'{class_id} 0.5 0.5 1.0 1.0\n')
        processed += 1
        show_progress(processed, total_images, f'{prefix} dataset')

  print()

convert_classification_dataset('data_folder/datasets/phase1/chinese/images', 'chn')
convert_classification_dataset('data_folder/datasets/phase1/mafood/MAFood121/images',  'maf')
print('ChineseFoodNet and MAFood-121 done.')