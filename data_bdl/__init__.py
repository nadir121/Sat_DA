from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet
from data.sentinel_dataset import sentinelDataset
from data.sentinel_datas import sentinelDatas
from data.sentinel_dataset_nolbl import sentinelDataset_noLbl
#from data.sentinel_worldview_dataset import sentinel_worldviewDataset
#from data.sentinel_worldview_datas import sentinel_worldviewDatas
#from data.sentinel_worldview_dataset_nolbl import sentinel_worldviewDataset_noLbl
from data.wv2_dataset import WV2Dataset
from data.wv2_dataset_nolbl import WV2Dataset_noLbl
from data.wv2_dataset_source import WV2Dataset_source
from data.wv2_dataset_source_cycle import WV2Dataset_sourceC
#from data.wv2_dataset_source_cycle448 import WV2Dataset_sourceC448
from data.sentinel_dataset_cycle import sentinelDatasetC
from data.dg_dataset import DGDataset
from data.dg_dataset224 import DGDataset224
from data.dg_dataset_nolbl import DGDataset_noLbl
from data.dg_dataset_nolbl224 import DGDataset_noLbl224
from data.synthia_dataset import SYNDataSet
from data.cityscapes_dataset_label import cityscapesDataSetLabel
import numpy as np
from torch.utils import data

#IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
#IMG_MEANS = np.array((87.98424478, 113.23702202, 77.58985945), dtype=np.float32)
#IMG_MEAND = np.array((104.0976662, 96.66178473, 71.79267964), dtype=np.float32)

IMG_MEAN = np.array((122.6789, 116.6688, 104.0070), dtype=np.float32)
IMG_MEANS = np.array((77.5899, 113.2370, 87.9842), dtype=np.float32)
IMG_MEAND = np.array((71.7927, 96.6618, 104.0977), dtype=np.float32)
IMG_MEANP = np.array((65.0786, 78.6180, 56.1907), dtype=np.float32)
image_sizes = {'cityscapes': (1024,512), 'gta5': (1280, 720), 'sentinel': (224, 224), 'wv': (512, 512), 'dg': (612, 612), 'pl': (448, 448)}

def CreateSrcDataLoader(args): 
    if args.source == 'gta5':
        source_dataset = GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                                    crop_size=image_sizes['gta5'], mean=IMG_MEAN) 
    elif args.source == 'sentinel' or args.source == 'sentinelFI':
        source_dataset = sentinelDataset(args.data_dir, args.data_list, args.lbl_list, max_iters=args.num_steps * args.batch_size,
                                    crop_size=image_sizes['sentinel'], mean=IMG_MEANS) 
    elif args.source == 'worldview' or args.source == 'worldviewFI' or args.source == 'worldviewGR':
        source_dataset = WV2Dataset_source(args.data_dir, args.data_list, args.lbl_list, max_iters=args.num_steps * args.batch_size,
                                    crop_size=image_sizes['wv'], mean=IMG_MEAN) 
    elif args.source == 'worldview_cycle' or args.source == 'worldviewFI_cycle' or args.source == 'worldviewGR_cycle':
        source_dataset = WV2Dataset_sourceC(args.data_dir, args.data_list, args.lbl_list, max_iters=args.num_steps * args.batch_size,
                                    crop_size=image_sizes['wv'], mean=IMG_MEAN)
    #elif args.source == 'worldview_cycle448' or args.source == 'worldviewFI_cycle448':
        #source_dataset = WV2Dataset_sourceC448(args.data_dir, args.data_list, args.lbl_list, max_iters=args.num_steps * args.batch_size,
                                    #crop_size=image_sizes['pl'], mean=IMG_MEANP)
    elif args.source == 'sentinelFI_cycle' or args.source == 'sentinel_cycle' or args.source == 'sentinelGR_cycle':
        source_dataset = sentinelDatasetC(args.data_dir, args.data_list, args.lbl_list, max_iters=args.num_steps * args.batch_size,
                                    crop_size=image_sizes['sentinel'], mean=IMG_MEANS)  
    elif args.source == 'deepglobe':
        source_dataset = DGDataset(args.data_dir, args.data_list, args.lbl_list, max_iters=args.num_steps * args.batch_size,
                                    crop_size=image_sizes['dg'], mean=IMG_MEAND)
    elif args.source == 'deepglobe224':
        source_dataset = DGDataset224(args.data_dir, args.data_list, args.lbl_list, max_iters=args.num_steps * args.batch_size,
                                    crop_size=image_sizes['sentinel'], mean=IMG_MEAND)  
    #elif args.source == 'sentinel_worldview':
        #source_dataset = sentinel_worldviewDataset(args.data_dir, args.data_list, args.lbl_list, max_iters=args.num_steps * args.batch_size)                             
    else:
        raise ValueError('The target dataset must be either gta5 or sentinel')
    
    source_dataloader = data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    return source_dataloader


def CreateTrgDataLoader(args):
    if args.target == 'sentinelGR' or args.target == 'sentinelFI' or args.target == 'sentinel' or args.target == 'sentinelc':
      if args.data_label_folder_target is not None:
          target_dataset = sentinelDatas(args.data_dir_target, args.data_list_target, max_iters=args.num_steps * args.batch_size,
                                           crop_size=image_sizes['sentinel'], mean=IMG_MEANS, set=args.set, label_folder=args.data_label_folder_target) 
      else:
          if args.set == 'train':
              target_dataset = sentinelDataset_noLbl(args.data_dir_target, args.data_list_target, max_iters=args.num_steps * args.batch_size,
                                               crop_size=image_sizes['sentinel'], mean=IMG_MEANS, set=args.set)
          else:
              target_dataset = sentinelDataset_noLbl(args.data_dir_target, args.data_list_target,
                                                crop_size=image_sizes['sentinel'], mean=IMG_MEANS, set=args.set)             
      if args.set == 'train':
          target_dataloader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
      else:
          target_dataloader = data.DataLoader(target_dataset, batch_size=1, shuffle=False, pin_memory=True)
      return target_dataloader

    elif args.target == 'deepglobe':
      if args.data_label_folder_target is not None:
          target_dataset = DGDataset(args.data_dir_target, args.data_list_target, max_iters=args.num_steps * args.batch_size,
                                           crop_size=image_sizes['dg'], mean=IMG_MEAND, set=args.set, label_folder=args.data_label_folder_target) 
      else:
          if args.set == 'train':
              target_dataset = DGDataset_noLbl(args.data_dir_target, args.data_list_target, max_iters=args.num_steps * args.batch_size,
                                               crop_size=image_sizes['dg'], mean=IMG_MEAND, set=args.set)
          else:
              target_dataset = DGDataset_noLbl(args.data_dir_target, args.data_list_target,
                                                crop_size=image_sizes['dg'], mean=IMG_MEAND, set=args.set)             
      if args.set == 'train':
          target_dataloader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
      else:
          target_dataloader = data.DataLoader(target_dataset, batch_size=1, shuffle=False, pin_memory=True)
      return target_dataloader

    elif args.target == 'deepglobe224':
      if args.data_label_folder_target is not None:
          target_dataset = DGDataset224(args.data_dir_target, args.data_list_target, max_iters=args.num_steps * args.batch_size,
                                           crop_size=image_sizes['sentinel'], mean=IMG_MEAND, set=args.set, label_folder=args.data_label_folder_target) 
      else:
          if args.set == 'train':
              target_dataset = DGDataset_noLbl224(args.data_dir_target, args.data_list_target, max_iters=args.num_steps * args.batch_size,
                                               crop_size=image_sizes['sentinel'], mean=IMG_MEAND, set=args.set)
          else:
              target_dataset = DGDataset_noLbl224(args.data_dir_target, args.data_list_target,
                                                crop_size=image_sizes['sentinel'], mean=IMG_MEAND, set=args.set)             
      if args.set == 'train':
          target_dataloader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
      else:
          target_dataloader = data.DataLoader(target_dataset, batch_size=1, shuffle=False, pin_memory=True)
      return target_dataloader

    elif args.target == 'worldview' or args.target == 'worldviewFI' or args.target == 'worldviewGR' or args.target == 'worldviewFIc' or args.target == 'worldviewc':
      if args.data_label_folder_target is not None:
          target_dataset = WV2Dataset(args.data_dir_target, args.data_list_target, max_iters=args.num_steps * args.batch_size,
                                           crop_size=image_sizes['wv'], mean=IMG_MEAN, set=args.set, label_folder=args.data_label_folder_target) 
      else:
          if args.set == 'train':
              target_dataset = WV2Dataset_noLbl(args.data_dir_target, args.data_list_target, max_iters=args.num_steps * args.batch_size,
                                               crop_size=image_sizes['wv'], mean=IMG_MEAN, set=args.set)
          else:
              target_dataset = WV2Dataset_noLbl(args.data_dir_target, args.data_list_target,
                                                crop_size=image_sizes['wv'], mean=IMG_MEAN, set=args.set)             
      if args.set == 'train':
          target_dataloader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
      else:
          target_dataloader = data.DataLoader(target_dataset, batch_size=1, shuffle=False, pin_memory=True)
      return target_dataloader

    elif args.target == 'pleiadesFI' or args.target == 'pleiades':
      if args.data_label_folder_target is not None:
          target_dataset = WV2Dataset(args.data_dir_target, args.data_list_target, max_iters=args.num_steps * args.batch_size,
                                           crop_size=image_sizes['pl'], mean=IMG_MEANP, set=args.set, label_folder=args.data_label_folder_target) 
      else:
          if args.set == 'train':
              target_dataset = WV2Dataset_noLbl(args.data_dir_target, args.data_list_target, max_iters=args.num_steps * args.batch_size,
                                               crop_size=image_sizes['pl'], mean=IMG_MEANP, set=args.set)
          else:
              target_dataset = WV2Dataset_noLbl(args.data_dir_target, args.data_list_target,
                                                crop_size=image_sizes['pl'], mean=IMG_MEANP, set=args.set)             
      if args.set == 'train':
          target_dataloader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
      else:
          target_dataloader = data.DataLoader(target_dataset, batch_size=1, shuffle=False, pin_memory=True)
      return target_dataloader



def CreateTrgDataSSLLoader(args):
    target_dataset = WV2Dataset_noLbl(args.data_dir_target, args.data_list_target,
                                           crop_size=image_sizes['wv'], mean=IMG_MEAN, set=args.set)
    target_dataloader = data.DataLoader(target_dataset, batch_size=1, shuffle=False, pin_memory=True)  
    return target_dataloader













