This repository is a course project of Computer Intelligence Lab, ETH Zurich, Fall 2022.
The project is a collaborative work from [Nicolas Kupper](https://github.com/Sinsho), [Noemi Marty](https://github.com/octaryne), [Nishendra Singh](https://github.com/nishendra3), [Xiao Wu](https://github.com/Adamink).

## Core files
- Datasets
    - [cil_512x512.py](configs/_base_/datasets/cil_512x512.py)
- Models
    - [Swin-Large](configs/swin/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_cil_noweight.py)
    - [Swin-Base](configs/swin/upernet_swin_base_patch4_window12_512x512_160k_cil_pretrain_384x384_22K_noweight.py)
    - [Swin-Small](configs/swin/upernet_swin_small_patch4_window7_512x512_160k_cil_pretrain_224x224_1K.py)
    - [PSPNet](configs/pspnet/pspnet_r50-d8_512x512_80k_cil.py)

## Install dependencies on Euler
```sh
module load gcc/6.3.0 gcc/8.2.0 python_gpu/3.8.5
pip install mmcv-full # default mmcv-1.6.0
pip install mmsegmentation # default mmseg-0.26.0
```

## Install dependencies on other platforms
```sh
conda create --name=cuda11.0 pytorch=1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch -c nvidia
conda activate cuda11.0
pip install -U openmim
mim install mmcv-full==1.6.0
pip install mmsegmentation
pip install future tensorboard
```

All following commands are executed under `mmseg` directory.
## Prepare datasets
```sh
git clone git@github.com:Sinsho/cil-2022.git # download datasets
python extract_data.py # make training/val split
```
## Prepare pretrained models
Choose one of the following pretrained models. The default one is Swin-L, however it cannot fit on reguler Euler graphics card (1080/1080Ti). To train on Euler, choose Swin-S.
```sh
mkdir pretrained
# Swin-L
wget -P pretrained https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k_20220318_091743-9ba68901.pth
# Swin-B
wget -P pretrained https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth
# Swin-S
wget -P pretrained https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth
# PSPNet
wget -P pretrained/ https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x512_40k_voc12aug/pspnet_r50-d8_512x512_40k_voc12aug_20200613_161222-ae9c1b8c.pth
# BEiT-L
wget -P pretrained https://download.openmmlab.com/mmsegmentation/v0.5/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k/upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth
```
## Train
See [train.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/train.md) for more details.

Choose one of the following config files accordingly. 
```sh
# Swin-L
CONFIG_FILE="configs/swin/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_cil_noweight.py"
# Swin-B
CONFIG_FILE="configs/swin/upernet_swin_base_patch4_window12_512x512_160k_cil_pretrain_384x384_22K_noweight.py"
# Swin-S
CONFIG_FILE="configs/swin/upernet_swin_small_patch4_window7_512x512_160k_cil_pretrain_224x224_1K.py"
# PSPNet
CONFIG_FILE="configs/pspnet/pspnet_r50-d8_512x512_80k_cil.py"
# BEiT-B
CONFIG_FILE='configs/beit/upernet_beit-base_640x640_160k_cil_ms.py"
```

on other platform:
```sh
# not distributed
bash dist_train.sh ${CONFIG_FILE} 1

# distributed on 4 GPUs
bash dist_train.sh ${CONFIG_FILE} 4
```
on Euler:
```sh
# not distributed
bsub -n 1 -R "rusage[mem=8000,ngpus_excl_p=1]" "sh dist_train.sh ${CONFIG_FILE} 1" 

# distributed on 2 GPUs
bsub -W 24:00 -n 4 -R "rusage[mem=8000,ngpus_excl_p=2]" "sh dist_train.sh ${CONFIG_FILE} 2"

# distributed on 2 GPUs asking for better GPUs 
bsub -W 24:00 -n 4 -R "rusage[mem=8000,ngpus_excl_p=2]" -R "select[gpu_model0==NVIDIATITANRTX]" "sh dist_train.sh ${CONFIG_FILE} 2"
```

## Check training curves
```sh
tensorboard --logdir work_dirs # vscode can help forwarding it to a local port
```

## Create submission file
```sh
# change the checkpoint accordingly
CHECKPOINT_FILE="work_dirs/pspnet_r50-d8_512x512_80k_cil/best_mIoU_iter_42900.pth"
SHOW_DIR="data/annotations/test/"
```

on laptop:
```sh
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${SHOW_DIR} # generating mask outputs in data/annotations/test/
python mask_to_submission.py # generating submission csv file
```
on Euler:
```sh
bsub -n 1 -R "rusage[mem=8000,ngpus_excl_p=1]" "python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${SHOW_DIR}" # generating mask outputs in data/annotations/test/ 
python mask_to_submission.py # generating submission csv file
```
## Launch multiple jobs on a single machine
To fix error message saying RuntimeError: Address already in use, change the port in the following line.
```sh
bsub -W 24:00 -n 4 -R "rusage[mem=8000,ngpus_excl_p=2]" "PORT=29503 sh dist_train.sh ${CONFIG_FILE} 2"
```

## Resume training
```sh
CHECKPOINT_FILE="work_dirs/upernet_swin_base_patch4_window12_256x256_160k_cil_pretrain_384x384_22K/best_mIoU_iter_6500.pth"

bsub -W 24:00 -n 2 -R "rusage[mem=8000,ngpus_excl_p=2]" "sh dist_train.sh ${CONFIG_FILE} 2 --resume-from ${CHECKPOINT_FILE}"
```

## Reference
- [Using the batch system on euler](https://scicomp.ethz.ch/wiki/Using_the_batch_system)
- [MMSegmentation repository](https://github.com/open-mmlab/mmsegmentation)
- [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/)
- [MMSegmentation tutorial](https://github.com/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb)
- [PSPNet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet)
- [Swin Transformer](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/swin)
- [SwinV2](https://github.com/microsoft/Swin-Transformer)
