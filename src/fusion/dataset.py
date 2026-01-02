import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor
import torch
import matplotlib.pyplot as plt

class FusionDataset(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args
        self.image_processor = AutoImageProcessor.from_pretrained(self.args.img_encoder_path)  # 앞으로 빼기
        if 'clip' in self.args.img_encoder_path:
            self.image_processor.size['shortest_edge'] = 224
        elif 'dino' in self.args.img_encoder_path:
            self.image_processor.size['shortest_edge'] = 224
        else:
            pass
        self.ColorJitter_functions = {0: transforms.functional.adjust_brightness,
                                      1: transforms.functional.adjust_contrast,
                                      2: transforms.functional.adjust_saturation,
                                      3: transforms.functional.adjust_hue}

    def __len__(self):
        return len(self.data)

    # def transforms(self, source_img, target_img, pos_t_img):
    #     pass
    #     return source_img, target_img, pos_t_img

    def __getitem__(self, idx):
        dat = self.data[idx]
        source_img_filename = dat['source_image']
        target_img_filename = dat['target_image']

        source_img_path = os.path.join(self.args.root_path, source_img_filename)
        target_img_path = os.path.join(self.args.root_path, target_img_filename)

        source_img = Image.open(source_img_path)
        target_img = Image.open(target_img_path)

        pose_path = target_img_path.replace('img', 'pose_img')
        pos_t_img = Image.open(pose_path)

        # if self.args.phase == 'train':
        #     source_img, target_img, pos_t_img = self.transforms(source_img, target_img, pos_t_img)

        processed_source_img = (self.image_processor(images=source_img,
                                                     return_tensors="pt").pixel_values).squeeze(dim=0)
        processed_target_img = (self.image_processor(images=target_img,
                                                     return_tensors="pt").pixel_values).squeeze(dim=0)
        pos_t_img = pos_t_img.resize([224, 224], Image.BICUBIC)
        processed_target_pose = (self.image_processor(images=pos_t_img,
                                                      return_tensors="pt").pixel_values).squeeze(dim=0)

        ## dropout s_img for clip
        if random.random() < self.args.img_pose_drop_rate:
            processed_source_img = torch.zeros(processed_source_img.shape)
        ## dropout pos_img for clip
        if random.random() < self.args.img_pose_drop_rate:
            processed_target_pose = torch.zeros(processed_target_pose.shape)


        return dict(source_img=processed_source_img, target_img=processed_target_img, target_pose=processed_target_pose,
                    source_img_filename=source_img_filename, target_img_filename=target_img_filename)
