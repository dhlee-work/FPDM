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
gt_path = './dataset/deepfashion/original_img'
gt_list = glob.glob(f'{gt_path}/**/*.jpg', recursive=True)
gt_list = np.array(gt_list)



# ## A1
generate_path = './dataset/deepfashion/generate'
distorated_list = glob.glob(f'{generate_path}/**/*.png', recursive=True)
# np.random.shuffle(distorated_list)
# len(distorated_list)
# distorated_list = [i.replace('generate', 'generate_ab2') for i in distorated_list]
# real_path = ['_'.join(i.split('_')[:-1]).replace('generate_ab2', 'original_img') + '.jpg' for i in distorated_list]
real_path = ['_'.join(i.split('_')[:-1]).replace('generate', 'original_img') + '.jpg' for i in distorated_list]
# gt_path = './dataset/deepfashion/resized512_img'
# gt_list = glob.glob(f'{gt_path}/**/*.jpg', recursive=True)
real_unique_path = np.unique(real_path)
gt_train_list = gt_list[~np.isin(gt_list, real_unique_path)]




######### A2
# gt_path = './dataset/deepfashion/generate_CFLD'
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
# real_train_path = []



_img_size = (176, 256)


fid_pred_data = FidRealDeepFashion(distorated_list, _img_size[::-1])
fid_pred_loader = DataLoader(
    fid_pred_data,
    batch_size=256,
    num_workers=6,
    pin_memory=True
)
fid_real_data = FidRealDeepFashion(gt_train_list, _img_size[::-1])
fid_real_loader = DataLoader(
    fid_real_data,
    batch_size=256,
    num_workers=6,
    pin_memory=True
)


 # (352, 512) #(176, 256) (352, 512)

fid_val = fid.calculate_batch(fid_pred_loader, fid_real_loader) # (176, 256) (352, 512)
lpips_val = lpips_obj.calculate_from_disk(distorated_list, real_path, img_size=_img_size, sort=False)
REC = rec.calculate_from_disk(distorated_list, real_path, None,  img_size=_img_size, sort=False, debug=False)

fid_val = round(fid_val, 4)
lpips_val = round(float(lpips_val),4)
ssim = round(REC['ssim'][0],4)
ssim_256 = round(REC['ssim_256'][0],4)
psnr = round(REC['psnr'][0], 4)
print(f"FID:{fid_val}, LPIPS:{lpips_val}, ssim:{ssim}, ssim-256:{ssim_256}, psnr:{psnr}")


# FID:0, LPIPS:0.1418, ssim:0.7304, ssim-256:0.7383, psnr:18.4333
# FID:0, LPIPS:0.1436, ssim:0.734, ssim-256:0.7426, psnr:18.5407

# 'fid': 58.43184508976461, 'lpips': 0.2103121280670166, 'ssim': 0.6950225738815003, 'psnr': 16.78713112345808,
# FID:10.259, LPIPS:0.176, ssim:0.72, psnr:17.188
# FID:9.444, LPIPS:0.165, ssim:0.725, psnr:17.398
# FID:9.35, LPIPS:0.154, ssim:0.733, psnr:17.844xqx
# FID:0, LPIPS:0, ssim:0.731, psnr:18.022
# 9.479875 LPIPS:0.174, ssim:0.733, psnr:18.12
# SOTA FID:6.804, LPIPS:0.1519, ssim:0.7378, psnr:18.235
# SOTA FID:5.7377, LPIPS:0.1519, ssim:0.7291, ssim-256:0.7378, psnr:18.2389

# ablation 2 src + kpt
# FID:5.902414, LPIPS:0.1471, ssim:0.7352, ssim-256:0.7386, psnr:18.69
# FID:6.341941, LPIPS:0.1741, ssim:0.7049, ssim-256:0.747, psnr:18.073


# ablation 3 src + kpt + global
# FID:0, LPIPS:0.1483, ssim:0.7357, ssim-256:0.7385, psnr:18.7313 -ep 60



# ablation 4 src + kpt + global + patch
# FID:0, LPIPS:0.1502, ssim:0.7272, ssim-256:0.7355, psnr:18.3332 Epoch-60 (FID:0, LPIPS:0.1289, ssim:0.7277, ssim-256:0.7475, psnr:18.3396
# FID:5.447280, LPIPS:0.1506, ssim:0.7246, ssim-256:0.7304, psnr:18.4066 ablation 4 final



