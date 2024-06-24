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

fid_val = 0
lpips_val = 0
ssim = 0
psnr = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
fid = FID()
lpips_obj = LPIPS()
lpips_obj.model.to(device)

rec = Reconstruction_Metrics()

# real_path = './datasets/deepfashing/img'
gt_path = './dataset/deepfashion/generate'

distorated_list = glob.glob(f'{gt_path}/**/*.png', recursive=True)
np.random.shuffle(distorated_list)
distorated_list = distorated_list

real_path = ['_'.join(i.split('_')[:-1]).replace('generate','original_img') + '.jpg' for i in distorated_list]


#
_img_size = (176, 256) # (176, 256) (352, 512)

fid_val = fid.calculate_from_disk(distorated_list, real_path, img_size=_img_size) # (176, 256) (352, 512)
lpips_val = lpips_obj.calculate_from_disk(distorated_list, real_path, img_size=_img_size, sort=False)
REC = rec.calculate_from_disk(distorated_list, real_path, None,  img_size=_img_size, sort=False, debug=False)
#
fid_val = round(fid_val, 3)
lpips_val = round(float(lpips_val),3)
ssim = round(REC['ssim_256'][0],3)
psnr = round(REC['psnr'][0],3)
print(f"FID:{fid_val}, LPIPS:{lpips_val}, ssim:{ssim}, psnr:{psnr}")

# 'fid': 58.43184508976461, 'lpips': 0.2103121280670166, 'ssim': 0.6950225738815003, 'psnr': 16.78713112345808,
# FID:10.259, LPIPS:0.176, ssim:0.72, psnr:17.188
# FID:9.444, LPIPS:0.165, ssim:0.725, psnr:17.398
# FID:9.35, LPIPS:0.154, ssim:0.733, psnr:17.844xqx
# FID:0, LPIPS:0, ssim:0.731, psnr:18.022
# 9.479875 LPIPS:0.174, ssim:0.733, psnr:18.12