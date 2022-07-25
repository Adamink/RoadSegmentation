import os.path as osp
import numpy as np
from PIL import Image
import os
import mmcv
from glob import glob
from random import sample
import shutil

def extract_data(data_root, local_root):
    # val_ids = [4, 10, 33, 37, 54, 60, 90, 126, 134, 139]
    palette = [[0, 0, 0], [255, 255, 255]]

    VAL_SIZE = 10
    val_pths = sample(glob(osp.join(data_root, 'training/images/*')), VAL_SIZE)
    train_pths = [pth for pth in glob(osp.join(data_root, 'training/images/*')) if pth not in val_pths]
    
    if osp.exists(local_root):
        shutil.rmtree(local_root)

    train_img_dir = osp.join(local_root, 'images/training/')
    val_img_dir = osp.join(local_root, 'images/validation/')
    train_gt_dir = osp.join(local_root, 'annotations/training/')
    val_gt_dir = osp.join(local_root, 'annotations/validation/')
    test_img_dir = osp.join(local_root, 'images/test/')

    os.makedirs(train_img_dir)
    os.makedirs(val_img_dir)
    os.makedirs(train_gt_dir)
    os.makedirs(val_gt_dir)
    os.makedirs(test_img_dir)

    for pth in train_pths:
        basename = osp.basename(pth)
        new_pth = osp.join(train_img_dir, basename)
        shutil.copy(pth, new_pth)

        gt_pth = pth.replace('images', 'groundtruth')
        gt = Image.open(gt_pth).convert("P")
        gt.putpalette(np.array(palette, dtype=np.uint8))
        gt.save(osp.join(train_gt_dir, basename))
    
    for pth in val_pths:
        basename = osp.basename(pth)
        new_pth = osp.join(val_img_dir, basename)
        shutil.copy(pth, new_pth)

        gt_pth = pth.replace('images', 'groundtruth')
        gt = Image.open(gt_pth).convert("P")
        gt.putpalette(np.array(palette, dtype=np.uint8))
        gt.save(osp.join(val_gt_dir, basename))
    
    for pth in glob(osp.join(data_root, 'test/images/*')):
        basename = osp.basename(pth)
        new_pth = osp.join(test_img_dir, basename)
        shutil.copy(pth, new_pth)

def process_annotation(data_root, ann_dir, save_dir):
    palette = [[0, 0, 0], [255, 255, 255]]
    if not osp.exists(osp.join(data_root, save_dir)):
        os.makedirs(osp.join(data_root, save_dir))
    for file in mmcv.scandir(osp.join(data_root, ann_dir), suffix = '.png'):
        print(file)
        seg_img = Image.open(osp.join(data_root, ann_dir, file)).convert("P")
        seg_img.putpalette(np.array(palette, dtype=np.uint8))
        seg_img.save(osp.join(data_root, save_dir, file))

def calc_ratio(train_data_ann_dir):
    not_road_cnt = 0
    road_cnt = 0
    import sys

    np.set_printoptions(threshold=sys.maxsize)

    for file in mmcv.scandir(train_data_ann_dir, suffix = '.png'):
        img = Image.open(osp.join(train_data_ann_dir, file))
        pix = np.array(img).reshape(-1)
        nonzero = np.count_nonzero(pix)
        zero = len(pix) - nonzero
        not_road_cnt += zero
        road_cnt += nonzero

    all = not_road_cnt + road_cnt
    print('{}, {}'.format(road_cnt / all, not_road_cnt / all))

if __name__ == '__main__':
    extract_data('../data/', './data/')
