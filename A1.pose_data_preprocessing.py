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
        img_path = kpt_txts[i].replace('/pose/', '/resized_img/').replace('txt', 'jpg')
        img = cv2.imread(img_path)
        # h, w, c = img.shape
        pos_img = PK.get_label_tensor(kpt_txts[i], img, param)
        if not os.path.exists(os.path.dirname(pose_image_path)):
            os.makedirs(os.path.dirname(pose_image_path), exist_ok=True)
        pos_img.save(pose_image_path, format='JPEG', quality=100)

def make_market1501_pose_txt(save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for phase in ['train', 'test']:
        file_txt = os.path.join(dataset_dir, f'./annotations/market-annotation-{phase}.csv')
        data = np.loadtxt(file_txt, dtype=str, skiprows=1, delimiter=':')

        kk = 0
        for i in range(len(data)):
            dat = data[i]
            filename = dat[0]
            save_file_path = os.path.join(save_path, filename.replace('.jpg', '.txt'))
            if os.path.exists(save_file_path):
                continue
            y = eval(dat[1])
            x = eval(dat[2])
            xy = np.array([x, y]).transpose(1, 0)
            if np.sum(xy) == -36:
                kk += 1
                continue
            save_file_path = os.path.join(save_path, filename.replace('.jpg', '.txt'))
            np.savetxt(save_file_path, xy)


def run_preprcessing_market1501(kpt_txts, save_dir):
    param = {}
    param['offset'] = 0
    param['stickwidth'] = 2
    param['anno_width'] = 64
    param['anno_height'] = 128
    PK = ProcessingKeypoints()
    for i in tqdm.tqdm(range(len(kpt_txts))):
        pose_image_path = kpt_txts[i].replace('pose', save_dir).replace('.txt', '.jpg')
        img_path = kpt_txts[i].replace('/pose/', '/img/').replace('txt', 'jpg')
        img = cv2.imread(img_path)
        pos_img = PK.get_label_tensor(kpt_txts[i], img, param)
        if not os.path.exists(os.path.dirname(pose_image_path)):
            os.makedirs(os.path.dirname(pose_image_path), exist_ok=True)
        pos_img.save(pose_image_path, format='JPEG', quality=100)

def run_preprcessing_sign(kpt_txts, save_dir):
    param = {}
    param['offset'] = 0
    param['stickwidth'] = 4
    param['anno_width'] = 512
    param['anno_height'] = 512
    PK = ProcessingSignKeypoints()
    for i in tqdm.tqdm(range(len(kpt_txts))):
        pose_image_path = kpt_txts[i].replace('pose', 'pose_img').replace('.txt', '.jpg')
        img_path = kpt_txts[i].replace('/pose/', '/img/').replace('txt', 'jpg')
        img = cv2.imread(img_path)
        # h, w, c = img.shape
        pos_img = PK.get_label_tensor(kpt_txts[i], img, param)
        if not os.path.exists(os.path.dirname(pose_image_path)):
            os.makedirs(os.path.dirname(pose_image_path), exist_ok=True)
        pos_img.save(pose_image_path, format='JPEG', quality=100)


root_dir = 'dataset'
dataname = 'sign' #; 'market1501' deepfashion
dataset_type = 'test'

if dataname == 'deepfashion':
    save_dir = 'pose_img'
    dataset_dir = os.path.join(root_dir, dataname)
    save_path = os.path.join(root_dir, dataname, save_dir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    kpt_txts = glob.glob(os.path.join(dataset_dir, 'pose/**/*.txt'), recursive=True)
    run_preprcessing_deepfashion(kpt_txts, save_dir)


elif dataname == 'market1501':
    save_dir = 'pose'
    dataset_dir = os.path.join(root_dir, dataname)
    save_path = os.path.join(root_dir, dataname, save_dir)
    make_market1501_pose_txt(save_path)

    save_dir = 'pose_img'
    save_path = os.path.join(root_dir, dataname, save_dir)
    kpt_txts = glob.glob(os.path.join(dataset_dir, 'pose/**/*.txt'), recursive=True)
    run_preprcessing_market1501(kpt_txts, save_dir)

elif dataname == 'sign':
    if not dataset_type == 'test':
        save_dir = 'five_people/sample_100000/pose_img'
        dataset_dir = os.path.join(root_dir, 'multi/five_people/sample_100000')
        save_path = os.path.join(root_dir, 'multi', save_dir)
    else:
        save_dir = './one_video_test/J/CUSH11632A_A11/pose_img'
        dataset_dir = os.path.join(root_dir, './one_video_test/J/CUSH11632A_A11')
        save_path = os.path.join(root_dir, save_dir)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    kpt_txts = glob.glob(os.path.join(dataset_dir, 'pose/**/*.txt'), recursive=True)
    run_preprcessing_sign(kpt_txts, save_dir)
else:
    print('wrong data name chose deepfashion or market1501')





