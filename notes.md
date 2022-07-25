# Model caparison
## on ADE20k

Medium-size model
| methods | mIoU | mIoU(ms+flip) |
| ---- | ---- | ---- |
| Swin-B |  48.13 | 49.72 |
| ViTAdapter-B(AugReg-B)| 51.9 | 52.5 |
| BEiT-B | 53.08 | 53.84 | 

| methods | mIoU | mIoU(ms+flip) | Mem(G)
| ---- | ---- | ---- | ---- |
| BEiT-L | 56.33 | 56.84 | 22.64 |
| BEiT-B | 53.08 | 53.84 | 15.88 |
| Swin-L | 52.25 | 54.12 |  12.42 |
| ViT-Adapter-L(UperNet, BEiT)| 58.0 | 58.4	| - |
| ViT-Adapter-L(Mask2Former, BEiT) |  58.3 | 59.0 | - | 
# PSPNet
## R101 vs R50
roughly 0.5 improvement on mIoU
## Multiscale testing
according to [PSPNet](https://arxiv.org/pdf/1612.01105.pdf), yields better performance of roughly 1.0
## crop_size and image_size
According to [PSPNet](https://arxiv.org/pdf/1612.01105.pdf), large crop_size is good.

| dataset | resolution | image_scale | crop_size |
| ---- | ---- | ---- | ---- |
| ADE20k | 1650x2200(widthxheight) | 2048x512 | 512x512|
| Cityscapes | 2048x1024 | 2049x1025 | 769x769 |
| Pascal VOC 2012 | 500x332 | 2048x512 | 512x512 |
| PASCAL-Context Dataset | 375x500 | 520x520 | 480x480 |
| COCO-Stuff 10k | 640x~480 | 2048x512 | 512x512 |
## Batch size
16 is used in [PSPNet](https://arxiv.org/pdf/1612.01105.pdf).



