import torch
import torch.nn as nn
from torch.autograd import Variable
from options_bdl.test_options import TestOptions
from data_bdl import CreateTrgDataLoader
from PIL import Image
import json
import os.path as osp
import os
import numpy as np
from model import CreateModel

palette = [0, 0, 0, 0, 255, 255, 255, 255, 0, 255, 0, 255, 0, 255, 0, 0, 0, 255, 255, 255,  255]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def compute_mIoU(gt_dir, pred_dir, target, devkit_dir='', restore_from=''):
    with open(osp.join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))


    if target =='deepglobe' or target =='deepglobe224':
      image_path_list = osp.join(devkit_dir, 'val.txt')
      label_path_list = osp.join(devkit_dir, 'val.txt')
    elif target =='sentinelFI' or target =='worldviewFI' or target =='pleiadesFI':
      image_path_list = osp.join(devkit_dir, 'FI_sat_val.txt')
      label_path_list = osp.join(devkit_dir, 'FI_label_val.txt')
    elif target =='sentinel' or target =='worldview':
      image_path_list = osp.join(devkit_dir, 'FIGR_sat_val.txt')
      label_path_list = osp.join(devkit_dir, 'FIGR_label_val.txt')
    elif target =='sentinelGR' or target =='worldviewGR':
      image_path_list = osp.join(devkit_dir, 'GR_sat_val.txt')
      label_path_list = osp.join(devkit_dir, 'GR_label_val.txt')
    elif target =='worldviewFIc':
      image_path_list = osp.join(devkit_dir, 'FIc.txt')
      label_path_list = osp.join(devkit_dir, 'FI_labelc.txt')
    elif target =='worldviewc':
      image_path_list = osp.join(devkit_dir, 'FIGR_satc_val.txt')
      label_path_list = osp.join(devkit_dir, 'FIGR_labelc_val.txt')
    elif target =='sentinelc':
      image_path_list = osp.join(devkit_dir, 'WNN_satc_val.txt')
      label_path_list = osp.join(devkit_dir, 'WNN_labelc_val.txt')
    else:
      image_path_list = osp.join(devkit_dir, 'sat_val.txt')
      label_path_list = osp.join(devkit_dir, 'label_val.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [osp.join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [osp.join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        if target =='deepglobe':
          pred = np.array(Image.open(os.path.splitext(pred_imgs[ind])[0]+'.png'))
          label = np.array(Image.open(devkit_dir+'/label/'+os.path.splitext(gt_imgs[ind])[0].split('/')[3] +'.png'))
        elif target =='deepglobe224':
          pred = np.array(Image.open(os.path.splitext(pred_imgs[ind])[0]+'.png').resize((224, 224), Image.BICUBIC))
          label = np.array(Image.open(devkit_dir+'/label/'+os.path.splitext(gt_imgs[ind])[0].split('/')[3] +'.png').resize((224, 224), Image.NEAREST))
        else:
          pred = np.array(Image.open(pred_imgs[ind]+'.png'))
          label = np.array(Image.open(gt_imgs[ind]+'.png'))
        label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            with open(restore_from+'_mIoU.txt', 'a') as f:
                f.write('{:d} / {:d}: {:0.2f}\n'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    hist2 = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        hist2[i] = hist[i] / np.sum(hist[i])
    
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        with open(restore_from+'_mIoU.txt', 'a') as f:
            f.write('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)) + '\n')
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    with open(restore_from+'_mIoU.txt', 'a') as f:
        f.write('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '\n')
    print('===> mIoU7: ' + str(round(np.nanmean(mIoUs) * 100, 2)))  
    
def main():
    opt = TestOptions()
    args = opt.initialize()
    if args.target=='deepglobe':
      size = 612
    elif args.target=='deepglobe224':
      size = 224
    elif args.target=='worldview' or args.target=='worldviewFI' or args.target=='worldviewGR' or args.target=='worldviewFIc' or args.target=='worldviewc':
      size = 512
    elif args.target=='sentinel' or args.target=='sentinelFI' or args.target=='sentinelGR' or args.target=='sentinelc':
      size = 224 
    if args.target=='pleiades' or args.target=='pleiadesFI' or args.target=='pleiadesGR':
      size = 448    
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
        
    model = CreateModel(args)
    
    
    model.eval()
    model.cuda()    
    targetloader = CreateTrgDataLoader(args)
    
    for index, batch in enumerate(targetloader):
        if index % 100 == 0:
            print ('%d processd' % index)
        image, _, name = batch
        output = model(Variable(image).cuda())
        output = nn.functional.softmax(output, dim=1)
        output = nn.functional.upsample(output, (size, size), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        output_nomask = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        output_col = colorize_mask(output_nomask)
        output_nomask = Image.fromarray(output_nomask)    
        name = name[0].split('/')[-1]
        if args.target=='deepglobe' or args.target=='deepglobe224':
          output_nomask.save('%s/%s.png' % (args.save, os.path.splitext(name)[0]))
        else:
          output_nomask.save('%s/%s.png' % (args.save, name))
        output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0])) 
        
    compute_mIoU(args.gt_dir, args.save, args.target, args.devkit_dir, args.restore_from)    

if __name__ == '__main__': 
    main()
    