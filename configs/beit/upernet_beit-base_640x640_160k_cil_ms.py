_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/cil_640x640.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

class_ratio=[0.822, 0.178]
class_weight=[1. / _ for _ in class_ratio]

load_from='pretrained/upernet_beit-base_8x2_640x640_160k_ade20k-eead221d.pth'

model = dict(
    # pretrained='pretrain/beit_base_patch16_224_pt22k_ft22k.pth',
    decode_head=dict(num_classes=2, loss_decode=dict(class_weight=class_weight)),
    auxiliary_head=dict(num_classes=2,loss_decode=dict(class_weight=class_weight)),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
log_config=dict(interval=50) # ?
evaluation=dict(interval=50, metric='mIoU', save_best='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=20000)
