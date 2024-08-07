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


gt_path = './dataset/deepfashion/vae_img'
distorated_list = glob.glob(f'{gt_path}/**/*.jpg', recursive=True)
np.random.shuffle(distorated_list)
len(distorated_list)
real_path = [i.replace('vae_img', 'resized512_img') for i in distorated_list]
# real_path = ['_'.join(i.split('_')[:-1]).replace('vae_img', 'resized512_img') + '.jpg' for i in distorated_list]
# gt_path = './dataset/deepfashion/resized512_img'
# gt_list = glob.glob(f'{gt_path}/**/*.jpg', recursive=True)
#
# path_dict = {}
# for i in range(len(gt_list)):
#     _id = gt_list[i].replace('./dataset/deepfashion/resized512_img','fashion').replace('id_', 'id').replace('/','').replace('_','')
#     path_dict[_id] = gt_list[i]
# real_path = []
# for i in range(len(distorated_list)):
#     ii = distorated_list[i].split('_2_')[-1].replace('_vis', '').replace('_', '').replace('.png','.jpg')
#     real_path.append(path_dict[ii])


#
_img_size = (352, 512)# (352, 512) # (176, 256) # (176, 256) #(176, 256) (352, 512)

fid_val = fid.calculate_from_disk(distorated_list, real_path, img_size=_img_size) # (176, 256) (352, 512)
lpips_val = lpips_obj.calculate_from_disk(distorated_list, real_path, img_size=_img_size, sort=False)
REC = rec.calculate_from_disk(distorated_list, real_path, None,  img_size=_img_size, sort=False, debug=False)
#['
fid_val = round(fid_val, 3)
lpips_val = round(float(lpips_val),3)
ssim = round(REC['ssim'][0],3)
ssim_256 = round(REC['ssim_256'][0],3)
psnr = round(REC['psnr'][0],3)
print(f"FID:{fid_val}, LPIPS:{lpips_val}, ssim:{ssim}, ssim-256:{ssim_256}, psnr:{psnr}")

# 'fid': 58.43184508976461, 'lpips': 0.2103121280670166, 'ssim': 0.6950225738815003, 'psnr': 16.78713112345808,
# FID:10.259, LPIPS:0.176, ssim:0.72, psnr:17.188
# FID:9.444, LPIPS:0.165, ssim:0.725, psnr:17.398
# FID:9.35, LPIPS:0.154, ssim:0.733, psnr:17.844xqx
# FID:0, LPIPS:0, ssim:0.731, psnr:18.022
# 9.479875 LPIPS:0.174, ssim:0.733, psnr:18.12
# SOTA FID 5.92, LPIPS 0.151, SSIM 0.74, PSNR 18.342 (PIDM : LPIPS:0.131, ssim:0.752, psnr:18.342)
# (176, 256) FID:6.088, LPIPS:0.167, ssim:0.728, psnr:17.885 ## deepfashion-diffusion-clip-init-wd00-lr1e-4-b32-cum8-size512-noaug-steplr/
# (352, 512) FID:6.72,  LPIPS:0.19,  ssim:0.749, psnr:17.553