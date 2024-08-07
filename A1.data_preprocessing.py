import glob
import json
import os

import cv2
import numpy as np
import tqdm
import shutil


# 데이터 train, test 명확하게 하기


def reformat_deepfashion_dataset(dataset_dir, image_list):
    image_dict = {}
    for i in range(len(image_list)):
        _path0 = image_list[i].replace(dataset_dir, '.')
        _path_key = image_list[i].replace('./dataset/deepfashion/img', '').replace('_', '').replace('/', '')
        image_dict[_path_key] = _path0

    filenames_train = []
    file_txt = '{}/annotations/fasion-pairs-train.csv'.format(dataset_dir)
    data = np.loadtxt(file_txt, dtype=str, skiprows=1)
    for i in data:
        source = i.split(',')[0]
        target = i.split(',')[1]
        source = source.replace('fashion', '').replace('_', '')
        target = target.replace('fashion', '').replace('_', '')
        source_path = image_dict[source]
        target_path = image_dict[target]
        filenames_train.append({'source_image': source_path,
                                'target_image': target_path})

    filenames_test = []
    file_txt = '{}/annotations/fasion-pairs-test.csv'.format(dataset_dir)
    data = np.loadtxt(file_txt, dtype=str, skiprows=1)
    for i in data:
        source = i.split(',')[0]
        target = i.split(',')[1]
        source = source.replace('fashion', '').replace('_', '')
        target = target.replace('fashion', '').replace('_', '')
        source_path = image_dict[source]
        target_path = image_dict[target]
        filenames_test.append({'source_image': source_path,
                               'target_image': target_path})
    return filenames_train, filenames_test

def reformat_market_dataset(dataset_dir):
    filenames_train = []
    file_txt = '{}/annotations/market-pairs-train.csv'.format(dataset_dir)
    data = np.loadtxt(file_txt, dtype=str, skiprows=1)
    for i in data:
        source = i.split(',')[0]
        target = i.split(',')[1]
        source = os.path.join('./img', source)
        target = os.path.join('./img', target)
        if not os.path.exists(os.path.join(dataset_dir, source)):
            print(f'warnning : no such a image {source}')
        if not os.path.exists(os.path.join(dataset_dir, source)):
            print(f'warnning : no such a image {target}')
        filenames_train.append({'source_image': source,
                                'target_image': target})
    len(filenames_train)

    filenames_test = []
    file_txt = '{}/annotations/market-pairs-test.csv'.format(dataset_dir)
    data = np.loadtxt(file_txt, dtype=str, skiprows=1)
    for i in data:
        source = i.split(',')[0]
        target = i.split(',')[1]
        source = os.path.join('./img', source)
        target = os.path.join('./img', target)
        if not os.path.exists(os.path.join(dataset_dir, source)):
            print(f'warnning : no such a image {source}')
        if not os.path.exists(os.path.join(dataset_dir, source)):
            print(f'warnning : no such a image {target}')
        filenames_test.append({'source_image': source,
                               'target_image': target})
    return filenames_train, filenames_test


def reformat_sign_dataset(dataset_dir):
    filenames_train = []
    file_txt = '{}/train_pairs.txt'.format(dataset_dir)
    if os.path.exists(file_txt):
        data = np.loadtxt(file_txt, dtype=str, skiprows=1)
        dname = dataset_dir.split('/')[-1]
        for i in data:
            source = i.split(',')[0].replace(f'./{dname}/', './')
            target = i.split(',')[1].replace(f'./{dname}/', './')
            # source = os.path.join('./img', source)
            # target = os.path.join('./img', target)
            if not os.path.exists(os.path.join(dataset_dir, source)):
                print(f'warnning : no such a image {source}')
            if not os.path.exists(os.path.join(dataset_dir,  target)):
                print(f'warnning : no such a image {target}')
            filenames_train.append({'source_image': source,
                                    'target_image': target})
        len(filenames_train)
    else:
        print('train file is not exists')

    filenames_test = []
    file_txt = '{}/test_pairs.txt'.format(dataset_dir)
    if os.path.exists(file_txt):
        data = np.loadtxt(file_txt, dtype=str, skiprows=1)
        dname = '/'.join(dataset_dir.split('/')[2:])
        for i in data:
            source = i.split(',')[0].replace(f'./{dname}/','./')
            target = i.split(',')[1].replace(f'./{dname}/','./')
            # source = os.path.join('./img', source)
            # target = os.path.join('./img', target)
            if not os.path.exists(os.path.join(dataset_dir, source)):
                print(f'warnning : no such a image {source}')
            if not os.path.exists(os.path.join(dataset_dir,  target)):
                print(f'warnning : no such a image {target}')
            filenames_test.append({'source_image': source,
                                   'target_image': target})
    else:
        print('test file is not exists')

    filenames_annotation = []
    file_txt = '{}/annotations.txt'.format(dataset_dir)
    if os.path.exists(file_txt):
        data = np.loadtxt(file_txt, dtype=str, skiprows=1)
        dname = '/'.join(dataset_dir.split('/')[2:])
        target_old = None
        for idx , i in enumerate(data):
            # if len(data) == (idx-1):
            #     continue
            # if idx == 0:
            #     source = i.split(',')[0].replace(f'./{dname}/','./')
            # else:
            #     source = target_old
            source = i.split(',')[0].replace(f'./{dname}/', './')
            target = i.split(',')[1].replace(f'./{dname}/','./')
            # target_old = target
            if not os.path.exists(os.path.join(dataset_dir, source)):
                print(f'warnning : no such a image {source}')
            if not os.path.exists(os.path.join(dataset_dir,  target)):
                print(f'warnning : no such a image {target}')
            filenames_annotation.append({'source_image': source,
                                   'target_image': target})
    else:
        print('video file is not exists')
    return filenames_train, filenames_test, filenames_annotation


def omit_image_only(_dataset, type):
    intr_path = []
    idx_bool = []
    for i in range(len(_dataset)):
        flag = True
        for j in ['source_image', 'target_image']:
            _path = _dataset[i][j]
            _path = _path.replace('img','pose_img')
            if not os.path.exists(os.path.join(dataset_dir, _path)):
                # print(f'warning : no pose matched with img : {_path}')
                intr_path.append(_path)
                flag = False
        if flag:
            idx_bool.append(True)
        else:
            idx_bool.append(False)
    print(f'omit {type}, {len(set(intr_path))} images, {len(idx_bool)-sum(idx_bool)}/{len(idx_bool)} pairs')
    _dataset = list(np.array(_dataset)[idx_bool])
    return _dataset

def resize_image(image_list, resized_dirname, ratio):
    for i in tqdm.tqdm(range(len(image_list))):
        c_path = image_list[i]
        s_path = c_path.replace('original_img', resized_dirname)
        if os.path.exists(s_path):
            continue
        img = cv2.imread(c_path)
        h, w, c = img.shape
        # img = cv2.resize(img, (int(w*ratio), int(h*ratio)))
        img = cv2.resize(img, (int(768), int(768)))
        dir_path = os.path.split(s_path)[0]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        cv2.imwrite(s_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print('finishing the image resize process')


def run_preprcessing_deepfashion(resized_dirname, dataset_dir):
    print('image resize preprocessing')
    image_list = glob.glob(os.path.join(dataset_dir, 'original_img/**/*.jpg'), recursive=True)
    resize_image(image_list, resized_dirname, ratio=0.5)

    # print('annotation preprocessing')
    # train_dataset, test_dataset = reformat_deepfashion_dataset(dataset_dir, image_list)
    # # check if imgs has no pose annotation omit.
    # train_dataset = omit_image_only(train_dataset, 'train')
    # test_dataset = omit_image_only(test_dataset, 'test')
    #
    # with open(os.path.join(dataset_dir, f'train_pairs_data.json'), 'w') as f:
    #     json.dump(train_dataset, f)
    # with open(os.path.join(dataset_dir, f'test_pairs_data.json'), 'w') as f:
    #     json.dump(test_dataset, f)
    # print('process finished !! ')

def run_preprcessing_market1501(dataset_dir):
    image_list = glob.glob(os.path.join(dataset_dir, 'Market-1501-v15.09.15/**/*.jpg'), recursive=True)
    for i in range(len(image_list)):
        file_path = image_list[i]
        basename = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(dataset_dir, 'img', basename))

    train_dataset, test_dataset = reformat_market_dataset(dataset_dir)
    with open(os.path.join(dataset_dir, f'train_pairs_data.json'), 'w') as f:
        json.dump(train_dataset, f)
    with open(os.path.join(dataset_dir, f'test_pairs_data.json'), 'w') as f:
        json.dump(test_dataset, f)
    print('process finished !! ')

def run_preprcessing_sign(dataset_dir):
    train_dataset, test_dataset, video_dataset = reformat_sign_dataset(dataset_dir)
    if  train_dataset:
        with open(os.path.join(dataset_dir, f'train_pairs_data.json'), 'w') as f:
            json.dump(train_dataset, f)
    if test_dataset:
        with open(os.path.join(dataset_dir, f'test_pairs_data.json'), 'w') as f:
            json.dump(test_dataset, f)
    if video_dataset:
        with open(os.path.join(dataset_dir, f'video_pairs_data.json'), 'w') as f:
            json.dump(video_dataset, f)
    print('process finished !! ')

root_dir = './dataset'
dataset = 'deepfashion'# 'sign' #'market1501' # sign one_video_test
dataset_type = 'video_test' #'video_test
resized_dirname = 'resized768_img'
dataset_dir = os.path.join(root_dir, dataset)

if dataset == 'deepfashion':
    run_preprcessing_deepfashion(resized_dirname, dataset_dir)
elif dataset == 'market1501':
    run_preprcessing_market1501(dataset_dir)
elif dataset =='sign':
    if dataset_type == 'video_test':
        dataset_dir = os.path.join(root_dir, 'video_test/')
        dataset_dir_list = glob.glob(dataset_dir+'*')
        for i in dataset_dir_list:
            run_preprcessing_sign(i)
    else:
        dataset_dir = os.path.join(root_dir, 'multi')
        run_preprcessing_sign(dataset_dir)

else:
    print('wrong dataset name')
