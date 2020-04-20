

import os
import os.path as osp
import numpy as np
import random
#import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image

class DGDataset224(data.Dataset):
    def __init__(self, root, list_path, lbl_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.lbl_ids = [i_id.strip() for i_id in open(lbl_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.lbl_ids = self.lbl_ids * int(np.ceil(float(max_iters) / len(self.lbl_ids)))
        self.files = []
        self.set = set
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        namel = os.path.splitext(self.lbl_ids[index])[0] +'.png'
        image = Image.open(osp.join(self.root, "sat/%s" %name)).convert('RGB')
        label = Image.open(osp.join(self.root, "label/%s" %namel))
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        return image.copy(), label.copy(), np.array(size), name

