from src.metrics.metrics import FID, LPIPS, Reconstruction_Metrics, preprocess_path_for_deform_task
import torch
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import cv2
from torchvision import transforms
# crop = transforms.RandomResizedCrop([176,256])
# crop.get_params(torch.zeros(176,256), scale=(0.8, 1.1), ratio=(0.75, 1.33))
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class FidRealDeepFashion(Dataset):
    def __init__(self, img_paths, test_img_size):
        super().__init__()
        self.img_items = img_paths
        self.transform_test = transforms.Compose([
            transforms.Resize(test_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path = self.img_items[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return self.transform_test(img)

fid_val = 0
lpips_val = 0
ssim = 0
psnr = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
fid = FID()
lpips_obj = LPIPS()
lpips_obj.model.to(device)

rec = Reconstruction_Metrics()
# gt_path = './dataset/deepfashion/original_img'
# gt_list = glob.glob(f'{gt_path}/**/*.jpg', recursive=True)
# gt_list = np.array(gt_list)
#
# generate_dname = 'generate_ab3'
# generate_path = f'./dataset/deepfashion/{generate_dname}'
# distorated_list = glob.glob(f'{generate_path}/**/*.png', recursive=True)
# for i in distorated_list:
#     plt.imread(i)
#
# # ## A1
# generate_dname = 'generate_ab4'
# generate_path = f'./dataset/deepfashion/{generate_dname}'
# distorated_list = glob.glob(f'{generate_path}/**/*.png', recursive=True)
# np.random.shuffle(distorated_list)
# len(distorated_list)
# # distorated_list = [i.replace('generate', 'generate_ab2') for i in distorated_list]
# # real_path = ['_'.join(i.split('_')[:-1]).replace('generate_ab2', 'original_img') + '.jpg' for i in distorated_list]
# real_path = ['_'.join(i.split('_')[:-1]).replace(generate_dname, 'original_img') + '.jpg' for i in distorated_list]
# real_unique_path = np.unique(real_path)
# gt_train_list = gt_list[np.isin(gt_list, real_unique_path)]
# uniq_list = np.unique(np.array([''.join(i.split('/')[4:7]) for i in gt_train_list]))
# aa = np.array([''.join(i.split('/')[4:7]) for i in distorated_list])
# distorated_list = np.array(distorated_list)[~np.isin(aa, uniq_list)]
# distorated_list = distorated_list.tolist()
# real_path = ['_'.join(i.split('_')[:-1]).replace(generate_dname, 'original_img') + '.jpg' for i in distorated_list]


## A2
generate_dname = 'generate_ab5_20_0'
generate_path = f'./dataset/deepfashion/{generate_dname}'
distorated_list = glob.glob(f'{generate_path}/**/*.png', recursive=True)

path_dict = {}
gt_path = './dataset/deepfashion/original_img'  # original_img resized512_img
gt_list = glob.glob(f'{gt_path}/**/*.jpg', recursive=True)
for i in range(len(gt_list)):
    _id = gt_list[i].replace('./dataset/deepfashion/original_img', '').replace('id_', 'id').replace('/','').replace('_','') # + '.jpg'
    path_dict[_id] = gt_list[i]
real_path = []
for i in range(len(distorated_list)):
    ii = distorated_list[i].split('.jpg_2_')[-1].replace('_vis', '').replace('_', '').replace('.png', '')
    real_path.append(path_dict[ii])

# real_unique_path = np.unique(real_path)
gt_path_list = np.array(glob.glob(f'{gt_path}/**/*.jpg', recursive=True))
gt_train_list = gt_path_list[~np.isin(gt_path_list, real_path)] # ~ train

# train 에만 있는것
# gt_train_list2 = gt_path_list[~np.isin(gt_path_list, real_unique_path)]
# uniq_list = np.unique(np.array([''.join(i.split('/')[4:7]).replace('_','') for i in gt_train_list2]))
# aa = np.array([i.split('.jpg_2_')[-1].split('id_')[0].replace('_', '') + 'id' + \
#                i.split('.jpg_2_')[-1].split('id_')[1].split('_')[0][:-2] for i in distorated_list])
# distorated_list = np.array(distorated_list)[~np.isin(aa, uniq_list)]
# distorated_list = distorated_list.tolist()

######### A3
# generate_dname = 'generate_512_PCDM'
# generate_path = f'./dataset/deepfashion/{generate_dname}'
# distorated_list = glob.glob(f'{generate_path}/**/*.png', recursive=True)
#
# # distorated_list = [i.replace('generate_ab3_20', 'generate_ab1_20') for i in distorated_list]
#
# path_dict = {}
# gt_path = './dataset/deepfashion/original_img'  # original_img resized512_img
# gt_list = glob.glob(f'{gt_path}/**/*.jpg', recursive=True)
# for i in range(len(gt_list)):
#     _id = gt_list[i].replace('./dataset/deepfashion/original_img', '').replace('id_', 'id').replace('/','').replace('_','') # + '.jpg'
#     path_dict[_id] = gt_list[i]
# real_path = []
# for i in range(len(distorated_list)):
#     ii = distorated_list[i].split('_to_')[-1].replace('_', '').replace('.png', '.jpg')
#     real_path.append(path_dict[ii])
#
# real_unique_path = np.unique(real_path)
# gt_path_list = np.array(glob.glob(f'{gt_path}/**/*.jpg', recursive=True))
# gt_train_list = gt_path_list[~np.isin(gt_path_list, real_unique_path)]
# gt_train_list2 = gt_path_list[~np.isin(gt_path_list, real_unique_path)]
# uniq_list = np.unique(np.array([''.join(i.split('/')[4:7]).replace('_','') for i in gt_train_list2]))
# aa = np.array([i.split('_to_')[-1].split('id_')[0].replace('_', '') + 'id' + \
#                i.split('_to_')[-1].split('id_')[1].split('_')[0] for i in distorated_list])
# distorated_list = np.array(distorated_list)[~np.isin(aa, uniq_list)]
# distorated_list = distorated_list.tolist()

# gt_path = './dataset/deepfashion/generate_512_CFLD'
# distorated_list = glob.glob(f'{gt_path}/**/*.png', recursive=True)
# np.random.shuffle(distorated_list)
# len(distorated_list)
# # real_path = [i.replace('generate', 'resized512_img') for i in distorated_list]
# path_dict = {}
# gt_path = './dataset/deepfashion/original_img'  # original_img resized512_img
# gt_list = glob.glob(f'{gt_path}/**/*.jpg', recursive=True)
# for i in range(len(gt_list)):
#     _id = gt_list[i].replace('./dataset/deepfashion/original_img', 'fashion').replace('id_', 'id').replace('/','').replace('_','')
#     path_dict[_id] = gt_list[i]
# real_path = []
# for i in range(len(distorated_list)):
#     ii = distorated_list[i].split('_2_')[-1].replace('_vis', '').replace('_', '').replace('.png','.jpg')
#     real_path.append(path_dict[ii])
#
#
# real_unique_path = np.unique(real_path)
# gt_path_list = np.array(glob.glob(f'{gt_path}/**/*.jpg', recursive=True))
# gt_train_list = gt_path_list[np.isin(gt_path_list, real_unique_path)]
# gt_train_list = np.unique(gt_train_list)
# import json
# if isinstance('./dataset/deepfashion/train_pairs_data.json' , str):
#     data = json.load(open('./dataset/deepfashion/train_pairs_data.json' ))
#
# gt_path_list = []
# for i in data:
#     gt_path_list.append(i['target_image'].replace('./img', './dataset/deepfashion/original_img'))
# gt_train_list = gt_path_list # np.unique(gt_path_list)


# real_path = []
# uniq_list = np.unique(np.array([''.join(i.split('/')[4:7]).replace('_','') for i in gt_train_list]))
# aa = np.array([''.join(''.join(i.split('_2_')[-1].split('_')[:1]).replace('fashion','')) for i in distorated_list])
# distorated_list = np.array(distorated_list)[~np.isin(aa, uniq_list)]
# distorated_list = distorated_list.tolist()
# for i in range(len(distorated_list)):
#     ii = distorated_list[i].split('_2_')[-1].replace('_vis', '').replace('_', '').replace('.png','.jpg')
#     real_path.append(path_dict[ii])


# gt_path_list = np.array(glob.glob(f'{gt_path}/**/*.jpg', recursive=True))
# gt_train_list = gt_path_list[np.isin(gt_path_list, real_path)]
# len(gt_train_list)

_img_size = (352, 512)  #(352, 512) # (176, 256) #
fid_pred_data = FidRealDeepFashion(distorated_list, _img_size[::-1])
fid_pred_loader = DataLoader(
    fid_pred_data,
    batch_size=64,
    num_workers=6,
    pin_memory=True
)
fid_real_data = FidRealDeepFashion(gt_train_list, _img_size[::-1])
fid_real_loader = DataLoader(
    fid_real_data,
    batch_size=64,
    num_workers=6,
    pin_memory=True
)

 # (352, 512) #(176, 256) (352, 512)
fid_val = fid.calculate_batch(fid_pred_loader, fid_real_loader)
lpips_val = lpips_obj.calculate_from_disk(distorated_list, real_path, img_size=_img_size, batch_size=10, sort=False)
REC = rec.calculate_from_disk(distorated_list, real_path, None,  img_size=_img_size, sort=False, debug=False)

fid_val = round(fid_val, 4)
lpips_val = round(float(lpips_val),4)
ssim = round(REC['ssim'][0],4)
ssim_256 = round(REC['ssim_256'][0],4)
psnr = round(REC['psnr'][0], 4)
print(f"FID:{fid_val}, LPIPS:{lpips_val}, ssim:{ssim}, ssim-256:{ssim_256}, psnr:{psnr}")

# PCDMs
# SOTA FID:0, LPIPS:0.1572, ssim:0.7179, ssim-256:0.728, psnr:18.3854 (test : 5.900)
# SOTA FID:5.53, LPIPS:0.1729, ssim:0.7003, ssim-256:0.7471, psnr:18.0279 (test: 5.53)

# CFLD
# SOTA FID:6.804 (7.048418), LPIPS:0.1519, ssim:0.7378, psnr:18.235 (test : fid_distance 5.688355)
# SOTA FID:7.149 , LPIPS:0.1819, ssim:0.7478, psnr:17.645

# ablation n1 src
# FID: 7.327990, LPIPS:0.1555, ssim:0.7251, ssim-256:0.7351, psnr:18.3538 (test : fid 5.752257)
# FID:7.6068, LPIPS:0.1824, ssim:0.6975, ssim-256:0.7443, psnr:17.7675

# ablation 1 src
# FID:7.627502, LPIPS:0.1472, ssim:0.734, ssim-256:0.7381, psnr:18.6994
# FID:7.8832, LPIPS:0.1744, ssim:0.7038, ssim-256:0.7464, psnr:18.081

# new
# FID:7.3288, LPIPS:0.1464, ssim:0.7357, ssim-256:0.7386, psnr:18.7442

# ablation 2 src + kpt
# FID:7.630600, LPIPS:0.147, ssim:0.7352, ssim-256:0.7386, psnr:18.69 (test : fid 5.481010)
# FID:7.924412, LPIPS:0.174, ssim:0.7049, ssim-256:0.747, psnr:18.073

# // uniq : FID:0, LPIPS:0.1608, ssim:0.704, ssim-256:0.7252, psnr:18.0392

# new
# FID:7.3446, LPIPS:0.1468, ssim:0.7357, ssim-256:0.7388, psnr:18.7008

# ablation 3 src + kpt + global
# FID: 7.237894, LPIPS:0.1457, ssim:0.7369, ssim-256:0.7387, psnr:18.7497 (test : fid 5.490529)
# FID:7.4791, LPIPS:0.1729, ssim:0.7053, ssim-256:0.7466, psnr:18.1237

# new 3
# FID:7.2906, LPIPS:0.1458, ssim:0.7372, ssim-256:0.7391, psnr:18.7659 (5.526802)
# FID:7.5368, LPIPS:0.173, ssim:0.7057, ssim-256:0.7471, psnr:18.1404 (5.916813)



# ablation 4 src + kpt + global + patch
# FID:0, LPIPS:0.1502, ssim:0.7272, ssim-256:0.7355, psnr:18.3332 Epoch-60 (FID:0, LPIPS:0.1289, ssim:0.7277, ssim-256:0.7475, psnr:18.3396
# FID:5.447280, LPIPS:0.1506, ssim:0.7246, ssim-256:0.7304, psnr:18.4066 ablation 4 final

# Unseen image

# ablation 5 src  + global

# ab1 _a
# FID:7.3527, LPIPS:0.1465, ssim:0.7357, ssim-256:0.7386, psnr:18.743

# ab2 _a
# FID:7.3446, LPIPS:0.1468, ssim:0.7357, ssim-256:0.7388, psnr:18.7008