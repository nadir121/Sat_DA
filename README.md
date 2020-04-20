# Sat_DA
 Improving land cover segmentation across satellites using domain adaptation
Implementation of [land cover segmentation across satellites using domain adaptation](https://arxiv.org/pdf/1912.05000.pdf) using [pytorch](http://pytorch.org/).

### Requirements

- NVIDIA GPU with 16GB + CUDA CuDNN.
- *python 3*,*pytorch 0.4.0*

### Datasets
* Worldview-2 dataset cannot be shared. feel free to request access to it via [ESA TPM](https://earth.esa.int/web/guest/data-access/browse-data-products/-/article/worldview-2-european-cities-dataset)
* Download the [Sentinel Dataset](www.kaggle.com/dataset/9b998f8430a4cb00d799ba4239ceec2c0202eb7b09b28470d5663b3b01a2fac6) as source dataset
* Download the [DeepGlobe Dataset](https://competitions.codalab.org/competitions/18468) as target dataset. You can access the data after signing up.
* Building datasets for CycleGAN can be done using: ``` create_datasets.py ```
* The Pleiades data can be purshased through [Airbus OneAtlas](https://oneatlas.airbus.com/living-library/subscription)


### Training
* The initial weights for BDL are available [here](https://drive.google.com/open?id=1TIrTmFKqEyf3pOKniv8-53m3v9SyBK0u)
* The model needs to be trained using the modified [BDL](https://github.com/liyunsheng13/BDL) first:
```
python -W ignore BDL.py --snapshot-dir ./snapshots/sen2dg \
              --source 'sentinel' \
              --target 'deepglobe' \
              --batch-size 2 \
              --init-weights ./init_weights/DeepLab_init.pth \
              --num-steps-stop 250000 \
              --model DeepLab \
              --data-dir ./dataset/Sentinel/ \
              --data-dir-target ./dataset/DG/ \
              --data-list ./dataset/Sentinel/FIGR_sat.txt \
              --lbl_list ./dataset/Sentinel/FIGR_label.txt \
              --data-list-target ./dataset/DG/train.txt \
              --label_list_target ./dataset/DG/train.txt
```
* To use DeeplabV3+ apply: ```--model DeepLabv3p ```
* The next step is translationg using a modified [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix):
```
python -W ignore train.py --dataroot './datasets/cyclegan/sentinel_deepglobe' \
                          --name 'sen2dg' \
                          --batch_size 2 \
                          --load_size 224 \
                          --crop_size 224 \
                          --lambda_semantic 2 \
                          --lambda_pa 1 \
                          --lambda_pb 0.5 \
                          --lambda_d 100 \
                          --init_weights './snapshots/sen2dg/sentinel_250000.pth'
```
* The next step is performing a training on the modified [BDL](https://github.com/liyunsheng13/BDL) again with the resulting translated data:
```
python -W ignore BDL.py --snapshot-dir ./snapshots/sen2dg_cycle \
              --source 'sentinel_cycle' \
              --target 'deepglobe' \
              --batch-size 2 \
              --init-weights ./init_weights/DeepLab_init.pth \
              --num-steps-stop 250000 \
              --model DeepLab \
              --data-dir ./results/ \
              --data-dir-target ./dataset/DG/ \
              --data-list ./results/sen2dg_sat.txt \
              --lbl_list ./results/sen2dg_label.txt \
              --data-list-target ./dataset/DG/train.txt \
              --label_list_target ./dataset/DG/train.txt \
```


### Evaluation
* The translated data is obtained by using ``` test.py ```:
```
python -W ignore test.py --dataroot './datasets/cyclegan/sentinel_deepglobe/trainA' \
                          --name 'sen2dg' \
                          --results_dir './results/train' \
                          --direction AtoB \
                          --model_suffix _A \
                          --load_size 224 \
                          --crop_size 224
```
* Building data list for the translated data is done using: ``` results/create_datasets.py ```
* The evaluation of the segmentation model after translation is done as following:
```
python -W ignore evaluation.py --restore-from ./snapshots/sen2dg_cycle/sentinel_cycle_250000
                                --model DeepLab \
                                --target 'deepglobe' \
                                --save ./eval/sen2dg_cycle \
                                --gt_dir ./dataset/DG/ \
                                --devkit_dir ./dataset/DG/ \ 
                                --data-list-target ./dataset/DG/val.txt 
                                --data-dir-target ./dataset/DG/
```


### Acknowledgment
This code is modified version of [BDL](https://github.com/liyunsheng13/BDL) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

