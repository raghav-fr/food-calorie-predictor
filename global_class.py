import os
import pandas as pd
def get_classes(root):
  return sorted([d for d in os.listdir(root)
                 if os.path.isdir(os.path.join(root, d))])
df= pd.read_csv('data_folder/datasets/phase1/chinese/food_class.csv')
chn = df['English Name'].tolist()
f101  = get_classes('data_folder/datasets/phase1/food101/images')
mafd  = get_classes('data_folder/datasets/phase1/mafood/MAFood121/images')

all_classes = sorted(set(
  [c.lower().replace('_',' ') for c in f101 + chn + mafd]
))
print(f'Unique Phase 1 classes: {len(all_classes)}')

with open('data_folder/merged/phase1_classes.txt','w',encoding='utf-8') as f:
  f.write('\n'.join(all_classes))