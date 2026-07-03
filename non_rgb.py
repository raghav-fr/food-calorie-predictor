import glob
from PIL import Image
import shutil
from tqdm import tqdm

non_rgb = []
all_imgs = glob.glob('data_folder/datasets/**/*.jpg', recursive=True) + \
           glob.glob('data_folder/datasets/**/*.png', recursive=True)
for path in tqdm(all_imgs):
  try:
    img = Image.open(path)
    if img.mode != 'RGB':
      non_rgb.append({'path': path, 'mode': img.mode})
      # Auto-fix: convert to RGB
      img.convert('RGB').save(path)
  except:
    pass

print(f'Non-RGB fixed: {len(non_rgb)}')