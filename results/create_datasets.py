import os
import numpy as np
import shutil
from tqdm import tqdm



dataset_basepathtr = 'train/sen2dg/test_latest/images'

tr = open('sen2dg_sat.txt', 'a')
lis = os.listdir(dataset_basepathtr)
for line in tqdm(lis):
  check = line.find("real")
  if check==-1:
    line = line.split('_fake')[0]
    tr.write(dataset_basepathtr+'/'+line+'\n')
tr.close()




