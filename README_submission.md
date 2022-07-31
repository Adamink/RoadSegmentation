This repository is a course project of Computer Intelligence Lab, ETH Zurich, Fall 2022.
The project is a collaborative work from [Nicolas Kupper](https://github.com/Sinsho), [Noemi Marty](https://github.com/octaryne), [Nishendra Singh](https://github.com/nishendra3), [Xiao Wu](https://github.com/Adamink).

## Install dependencies on Euler
```sh
module load gcc/6.3.0 gcc/8.2.0 python_gpu/3.8.5
pip install mmcv-full # default mmcv-1.6.0
pip install mmsegmentation # default mmseg-0.26.0
```

## Prepare pretrained models
```sh
mkdir pretrained
# Swin-L
wget -P pretrained https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k_20220318_091743-9ba68901.pth
```
## Train
See [train.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/train.md) for more details.

Specify config file 
```sh
# Swin-L
CONFIG_FILE="configs/swin/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_cil_noweight.py"
```

on Euler:
```sh
bsub -W 24:00 -n 1 -R "rusage[mem=8000,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIATITANRTX]" "sh dist_train.sh ${CONFIG_FILE} 1"
```

## Create submission file
on Euler:
```sh
bsub -n 1 -R "rusage[mem=8000,ngpus_excl_p=1]" "python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${SHOW_DIR}" # generating mask outputs in data/annotations/test/ 
python mask_to_submission.py # generating submission csv file
```

## Reference
- [Using the batch system on euler](https://scicomp.ethz.ch/wiki/Using_the_batch_system)
- [MMSegmentation repository](https://github.com/open-mmlab/mmsegmentation)
- [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/)
- [MMSegmentation tutorial](https://github.com/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb)
- [Swin Transformer](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/swin)
- [SwinV2](https://github.com/microsoft/Swin-Transformer)
