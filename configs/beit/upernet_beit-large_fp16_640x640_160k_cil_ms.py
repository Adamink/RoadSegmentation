
_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/cil_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_320k.py'
]

class_ratio=[0.822, 0.178]
class_weight=[1. / _ for _ in class_ratio]
load_from='upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k_20220318_091743-9ba68901.pth'
model = dict(
    # pretrained='pretrain/beit_large_patch16_224_pt22k_ft22k.pth',
    backbone=dict(
        type='BEiT',
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        qv_bias=True,
        init_values=1e-6,
        drop_path_rate=0.2,
        out_indices=[7, 11, 15, 23]),
    neck=dict(embed_dim=1024, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024], num_classes=2, channels=1024, 
        loss_decode=dict(class_weight=class_weight)),
    auxiliary_head=dict(in_channels=1024, num_classes=2,
        loss_decode=dict(class_weight=class_weight)),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.95))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=1)
optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook', cumulative_iters=2)

fp16 = dict()
log_config=dict(interval=50) # ?
evaluation=dict(interval=50, metric='mIoU', save_best='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=20000)
