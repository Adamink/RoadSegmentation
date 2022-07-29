## Install
```sh
# conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda create --name=cuda11.0 pytorch=1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch -c nvidia
conda activate cuda11.0
# conda create --name=cuda11.1 pytorch=1.9.1 torchvision==0.10.1 cudatoolkit=11.1 cudnn=8.0. -c pytorch -c nvidia
# conda install nccl=2.7.8 -c conda-forge
# pip install setuptools==59.5.0

pip install -U openmim
mim install mmcv-full==1.6.0
pip install mmsegmentation
pip install future tensorboard
sudo apt-get install ffmpeg libsm6 libxext6 -y
```

## Prepare dataset
```sh
git clone git@github.com:Sinsho/cil-2022.git
python extract_data.py
```
## Prepare pretrained model

```sh
mkdir pretrained
# Swin-L
wget -P pretrained https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k_20220318_091743-9ba68901.pth

# Swin-B
wget -P pretrained https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth

# BEiT-L
wget -P pretrained https://download.openmmlab.com/mmsegmentation/v0.5/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k/upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth
```

## Run
```sh
# Swin-L
CONFIG_FILE="configs/swin/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_cil.py"
# Swin-B
CONFIG_FILE="configs/swin/upernet_swin_base_patch4_window12_512x512_160k_cil_pretrain_384x384_22K.py"
# Swin-B nopretrain
CONFIG_FILE="configs/swin/upernet_swin_base_patch4_window12_512x512_160k_cil_nopretrain.py"

# Beit-B
CONFIG_FILE="configs/beit/upernet_beit-base_640x640_160k_cil_ms.py"
# Swin-L noweight
CONFIG_FILE="configs/swin/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_cil_noweight.py"

CUDA_VISIBLE_DEVICES=0 bash dist_train.sh ${CONFIG_FILE} 1
```

## Test
```sh
CONFIG_FILE="configs/swin/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_cil.py"
CHECKPOINT_FILE="work_dirs/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_cil/best_mIoU_iter_76850.pth"
SHOW_DIR="data/annotations/test/"

CONFIG_FILE="configs/beit/upernet_beit-base_640x640_160k_cil_ms.py"
CHECKPOINT_FILE="work_dirs/upernet_beit-base_640x640_160k_cil_ms/best_mIoU_iter_55300.pth"

python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${SHOW_DIR} # generating mask outputs in data/annotations/test/
python mask_to_submission.py # generating submission csv file
```