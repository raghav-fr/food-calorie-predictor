import glob
from PIL import Image
import json
from tqdm import tqdm

bad = []
all_imgs = glob.glob('data_folder/datasets/**/*.jpg', recursive=True) + \
           glob.glob('data_folder/datasets/**/*.png', recursive=True)

for path in tqdm(all_imgs):
  try:
    img = Image.open(path)
    img.verify()
  except Exception as e:
    bad.append({'path': path, 'error': str(e)})

with open('data_folder/logs/corrupted_images.json','w') as f:
  json.dump(bad, f, indent=2)
print(f'Corrupted: {len(bad)} / {len(all_imgs)}')