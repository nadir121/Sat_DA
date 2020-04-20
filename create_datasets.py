#Creates datasets for CycleGAN

import os
import numpy as np
import shutil
from tqdm import tqdm

def pause():
    programPause = input("Press the <ENTER> key to continue...")


trainA = open('./dataset/Sentinel/FIGR_sat.txt')
trainB = open('./dataset/DG/train.txt')
valA = open('./dataset/Sentinel/FIGR_sat_val.txt')
valB = open('./dataset/DG/val.txt')

locA = './datasets/cyclegan/sentinel_deepglobe/trainA'
locB = './datasets/cyclegan/sentinel_deepglobe/trainB'
locAv = './datasets/cyclegan/sentinel_deepglobe/valA'
locBv = './datasets/cyclegan/sentinel_deepglobe/valB'

if not os.path.exists(locA):
  os.makedirs(locA)
if not os.path.exists(locB):
  os.makedirs(locB)
if not os.path.exists(locAv):
  os.makedirs(locAv)
if not os.path.exists(locBv):
  os.makedirs(locBv)


lines = trainA.readlines()
for line in tqdm(lines):
  line = line.strip()
  shutil.copy('./dataset/Sentinel/'+line+'.png', locA)
trainA.close()

lines = valA.readlines()
for line in tqdm(lines):
  line = line.strip()
  shutil.copy('./dataset/Sentinel/'+line+'.png', locAv)
valA.close()

lines = trainB.readlines()
for line in tqdm(lines):
  line = line.strip()
  shutil.copy('./dataset/DG/'+line+'.png', locB)
trainB.close()	

lines = valB.readlines()
for line in tqdm(lines):
  line = line.strip()
  shutil.copy('./dataset/DG/'+line+'.png', locBv)
valB.close()



