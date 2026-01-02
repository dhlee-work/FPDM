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

FID_real_dataset = 'test' # train or test
fid_val = 0
lpips_val = 0
ssim = 0
psnr = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
fid = FID()
lpips_obj = LPIPS()
lpips_obj.model.to(device)

rec = Reconstruction_Metrics()

## A2 generate_ab1_final_256 generate_ab1_final_256
generate_dname = 'generate_ab6_20_0' # generate_ab8_final_512_20_0'# 'generate_ab5-only512_final_512_20_0' #
generate_path = f'./dataset/deepfashion_samples/{generate_dname}'
distorated_list = glob.glob(f'{generate_path}/**/*.png', recursive=True)

path_dict = {}
gt_path = './dataset/deepfashion_samples/img'  # original_img resized512_img
gt_list = glob.glob(f'{gt_path}/**/*.jpg', recursive=True)
for i in range(len(gt_list)):
    _id = gt_list[i].replace('./dataset/deepfashion_samples/img', '').replace('id_', 'id').replace('/','').replace('_','') # + '.jpg'
    path_dict[_id] = gt_list[i]
real_path = []
for i in range(len(distorated_list)):
    ii = distorated_list[i].split('.jpg_2_')[-1].replace('_vis', '').replace('_', '').replace('.png', '')
    real_path.append(path_dict[ii])

# real_unique_path = np.unique(real_path)
gt_path_list = np.array(glob.glob(f'{gt_path}/**/*.jpg', recursive=True))
if FID_real_dataset == 'train':
    gt_train_list = gt_path_list[~np.isin(gt_path_list, real_path)] # ~ train
elif FID_real_dataset == 'test':
    gt_train_list = gt_path_list[np.isin(gt_path_list, real_path)] # ~ train
else:
    assert ('FID_real_dataset should be train or test')

_img_size = (352, 512) #  (176, 256) # (352, 512) # (176, 256) # (352, 512) # (176, 256) # (352, 512) # (176, 256) #(352, 512) # (176, 256) #
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
fid_val = fid.calculate_batch(fid_pred_loader, fid_real_loader) # (176, 256) (352, 512)
lpips_val = lpips_obj.calculate_from_disk(distorated_list, real_path, img_size=_img_size, batch_size=10, sort=False)
REC = rec.calculate_from_disk(distorated_list, real_path, None,  img_size=_img_size, sort=False, debug=False)

fid_val = round(fid_val, 4)
lpips_val = round(float(lpips_val),4)
ssim = round(REC['ssim'][0],4)
ssim_256 = round(REC['ssim_256'][0],4)
psnr = round(REC['psnr'][0], 4)
print(f"FID:{fid_val}, LPIPS:{lpips_val}, ssim:{ssim}, ssim-256:{ssim_256}, psnr:{psnr}")

