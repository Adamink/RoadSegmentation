## Core config files
- Datasets
    - [cil_256x256.py](configs/_base_/datasets/cil_256x256.py)
    - [cil_512x512.py](configs/_base_/datasets/cil_512x512.py)
- Models
    - [swin_base_256x256](configs/swin/upernet_swin_base_patch4_window12_256x256_160k_cil_pretrain_384x384_22K.py)
    - [swin_small_512x512](configs/swin/upernet_swin_small_patch4_window7_512x512_160k_cil_pretrain_224x224_1K.py)
    - [pspnet_512x512](configs/pspnet/pspnet_r50-d8_512x512_80k_cil.py)
    - [pspnet_256x256](configs/pspnet/pspnet_r50-d8_256x256_80k_cil.py)

## Install dependencies
```sh
module load gcc/6.3.0 gcc/8.2.0 python_gpu/3.8.5
pip install mmcv-full # default mmcv-1.6.0
pip install mmsegmentation # default mmseg-0.26.0
```

All following commands are executed under `mmseg` directory.
## Prepare datasets
```sh
python extract_data.py
```
## Prepare pretrained models
Choose one of the following pretrained models.
```sh
# PSPNet 512x512 on VOC
wget -P pretrained/ https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x512_40k_voc12aug/pspnet_r50-d8_512x512_40k_voc12aug_20200613_161222-ae9c1b8c.pth

# Swin-B 512x512 on ADE20k
wget -P pretrained/ https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth

# Swin-S 512x512 on ADE20k
wget -P pretrained/ https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth
```
## Train
See [train.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/train.md) for more details.

Choose one of the following config files.
```sh 
CONFIG_FILE="configs/swin/upernet_swin_small_patch4_window7_512x512_160k_cil_pretrain_224x224_1K.py"
CONFIG_FILE="configs/swin/upernet_swin_base_patch4_window12_256x256_160k_cil_pretrain_384x384_22K.py"
CONFIG_FILE="configs/pspnet/pspnet_r50-d8_512x512_80k_cil.py"
```
on laptop:

```sh
sh dist_train.sh ${CONFIG_FILE} 1
```
on Euler:
```sh
# not distributed
bsub -n 1 -R "rusage[mem=8000,ngpus_excl_p=1]" "sh dist_train.sh ${CONFIG_FILE} 1" 

# distributed on 2 GPUS
bsub -W 24:00 -n 4 -R "rusage[mem=8000,ngpus_excl_p=2]" "sh dist_train.sh ${CONFIG_FILE} 2"

# distributed on 2 GPUS asking for better GPUs 
bsub -W 24:00 -n 4 -R "rusage[mem=8000,ngpus_excl_p=2]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" "sh dist_train.sh ${CONFIG_FILE} 2"
```

## Check training curves
```sh
tensorboard --logdir work_dirs/pspnet/tf_logs # vscode can help forwarding it to a local port
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
