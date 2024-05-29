import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor
import torch

class FusionDataset(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args
        # self.PK_module = ProcessingKeypoints()
        self.image_processor = AutoImageProcessor.from_pretrained(self.args.img_encoder_path)  # 앞으로 빼기
        self.image_processor.size['shortest_edge'] = 224

        self.ColorJitter_functions = {0: transforms.functional.adjust_brightness,
                                      1: transforms.functional.adjust_contrast,
                                      2: transforms.functional.adjust_saturation,
                                      3: transforms.functional.adjust_hue}

    def __len__(self):
        return len(self.data)

    def transforms(self, source_img, target_img, pos_t_img):
        # Random crop
        # if random.random() < 0.5:
        crop = transforms.RandomResizedCrop(self.args.img_size)
        params = crop.get_params(source_img, scale=(0.8, 1.1), ratio=(0.75, 1.33))
        source_img = transforms.functional.crop(source_img, *params)
        source_img = transforms.functional.resize(source_img, crop.size[::-1])

        # params = crop.get_params(source_img, scale=(0.8, 1.1), ratio=(0.75, 1.33))
        target_img = transforms.functional.crop(target_img, *params)
        target_img = transforms.functional.resize(target_img, crop.size[::-1])

        pos_t_img = transforms.functional.crop(pos_t_img, *params)
        pos_t_img = transforms.functional.resize(pos_t_img, crop.size[::-1])

        # Random horizontal flipping
        if random.random() < 0.0:
            source_img = transforms.functional.hflip(source_img)
        if random.random() < 0.0:
            target_img = transforms.functional.hflip(target_img)
            pos_t_img = transforms.functional.hflip(
                pos_t_img)  # plt.imshow(pos_t_img.numpy().transpose(1,2,0)); plt.show()

        if random.random() < 0.5:  # 0.8
            jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
            params = jitter.get_params(jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue)

            for i in np.array(params[0]):
                source_img = self.ColorJitter_functions[i](source_img, params[i + 1])
                target_img = self.ColorJitter_functions[i](target_img, params[i + 1])

        if random.random() < 0.1:  # 0.2
            target_img = transforms.functional.rgb_to_grayscale(target_img, num_output_channels=3)
            source_img = transforms.functional.rgb_to_grayscale(source_img, num_output_channels=3)

        return source_img, target_img, pos_t_img

    def __getitem__(self, idx):
        dat = self.data[idx]
        source_img_filename = dat['source_image']
        target_img_filename = dat['target_image']

        source_img_path = os.path.join(self.args.root_path, source_img_filename)
        target_img_path = os.path.join(self.args.root_path, target_img_filename)

        source_img = Image.open(source_img_path)
        source_img = source_img.resize(self.args.img_size)

        target_img = Image.open(target_img_path)
        target_img = target_img.resize(self.args.img_size)

        pose_path = target_img_path.replace('img', 'pose_img')
        pos_t_img = Image.open(pose_path)
        pos_t_img = pos_t_img.resize(self.args.img_size)


        if self.args.phase == 'train':
            source_img, target_img, pos_t_img = self.transforms(source_img, target_img, pos_t_img)


        processed_source_img = (self.image_processor(images=source_img.resize((224, 224),Image.BICUBIC),
                                                     return_tensors="pt").pixel_values).squeeze(dim=0)
        processed_target_img = (self.image_processor(images=target_img.resize((224, 224),Image.BICUBIC),
                                                     return_tensors="pt").pixel_values).squeeze(dim=0)
        processed_target_pose = (self.image_processor(images=pos_t_img.resize((224, 224),Image.BICUBIC),
                                                      return_tensors="pt").pixel_values).squeeze(dim=0)

        ## dropout s_img for clip
        if random.random() < self.args.img_pose_drop_rate:
            processed_source_img = torch.zeros(processed_source_img.shape)
        ## dropout pos_img for clip
        if random.random() < self.args.img_pose_drop_rate:
            processed_target_pose = torch.zeros(processed_target_pose.shape)


        return dict(source_img=processed_source_img, target_img=processed_target_img, target_pose=processed_target_pose)
