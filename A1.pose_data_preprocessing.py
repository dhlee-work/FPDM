import glob
import os

import cv2
import tqdm
import numpy as np
from src.fusion.datautil import ProcessingKeypoints, ProcessingSignKeypoints


def run_preprcessing_deepfashion(kpt_txts, save_dir):
    param = {}
    param['offset'] = 40
    param['stickwidth'] = 4
    param['anno_width'] = 176
    param['anno_height'] = 256
    PK = ProcessingKeypoints()
    for i in tqdm.tqdm(range(len(kpt_txts))):
        pose_image_path = kpt_txts[i].replace('pose', save_dir).replace('.txt', '.jpg')
        img_path = kpt_txts[i].replace('/pose/', '/img/').replace('txt', 'jpg')
        img = cv2.imread(img_path)
        pos_img = PK.get_label_tensor(kpt_txts[i], img, param)
        if not os.path.exists(os.path.dirname(pose_image_path)):
            os.makedirs(os.path.dirname(pose_image_path), exist_ok=True)
        pos_img.save(pose_image_path, format='JPEG', quality=100)


root_dir = 'dataset'
dataname = 'deepfashion_samples' #; 'market1501' deepfashion
dataset_type = 'train'

if dataname == 'deepfashion_samples':
    save_dir = 'pose_img'
    dataset_dir = os.path.join(root_dir, dataname)
    save_path = os.path.join(root_dir, dataname, save_dir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    kpt_txts = glob.glob(os.path.join(dataset_dir, 'pose/**/*.txt'), recursive=True)
    run_preprcessing_deepfashion(kpt_txts, save_dir)

else:
    print('wrong data name chose deepfashion')





