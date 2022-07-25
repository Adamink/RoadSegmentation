_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/cil_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
load_from='pretrained/pspnet_r50-d8_512x512_40k_voc12aug_20200613_161222-ae9c1b8c.pth'
class_ratio=[0.822, 0.178]
class_weight=[1. / _ for _ in class_ratio]
# modify num classes of the model in decode/auxiliary head
# modify weight of road and non-road part
model=dict(
    decode_head=dict(
        num_classes = 2,
        loss_decode=dict(class_weight=class_weight)
    ),
    auxiliary_head=dict(
        num_classes = 2,
        loss_decode=dict(class_weight=class_weight)
    )
)
log_config=dict(interval=50) # ?
evaluation=dict(interval=50, metric='mIoU', save_best='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=20000)
