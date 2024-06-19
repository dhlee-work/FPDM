import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor
import torch.nn.functional as F
from src.fusion.datautil import ProcessingKeypoints, ProcessingSignKeypoints
# from timm.data.random_erasing import RandomErasing
import matplotlib.pyplot as plt


def FPDM_Collate_fn(data):
    processed_source_image = torch.stack([example["processed_s_img"] for example in data])
    # processed_source_image = processed_source_image.to(memory_format=torch.contiguous_format).float()

    # processed_target_image = torch.stack([example["processed_t_img"] for example in data])
    # processed_target_image = processed_target_image.to(memory_format=torch.contiguous_format).float()

    processed_target_pose = torch.stack([example["processed_t_pose"] for example in data])
    # processed_pose_image = processed_pose_image.to(memory_format=torch.contiguous_format).float()

    # source_image = torch.stack([example["trans_s_img"] for example in data])
    # source_image = source_image.to(memory_format=torch.contiguous_format).float()

    target_pose = torch.stack([example["trans_t_pose"] for example in data])
    # target_pose = target_pose.to(memory_format=torch.contiguous_format).float()

    target_image = torch.stack([example["trans_t_img"] for example in data])
    # target_image = target_image.to(memory_format=torch.contiguous_format).float()

    return {
        "processed_source_image": processed_source_image,
        "processed_target_pose": processed_target_pose,
        # "source_image": source_image,
        "target_pose": target_pose,
        "target_image": target_image,
    }


class FPDM_Dataset(Dataset):
    def __init__(
            self,
            json_file,
            image_root_path,
            phase='train',
            src_encoder_path=None,
            model_img_size=(512, 512),
            img_size=(512, 512),
            imgs_drop_rate=0.0,
            pose_drop_rate=0.0,
            pose_erase_rate=0.02
    ):
        if isinstance(json_file, str):
            self.data = json.load(open(json_file))
        else:
            self.data = json_file
        self.image_root_path = image_root_path

        self.phase = phase
        self.img_size = img_size
        self.model_img_size = model_img_size
        self.imgs_drop_rate = imgs_drop_rate
        self.pose_drop_rate = pose_drop_rate
        self.pose_erase_rate = pose_erase_rate
        self.src_encoder_path = src_encoder_path
        self.image_processor = AutoImageProcessor.from_pretrained(self.src_encoder_path)  # 앞으로 빼기
        self.image_processor.size['shortest_edge'] = 224

        # self.image_src_processor = AutoImageProcessor.from_pretrained(self.src_encoder_path)  # 앞으로 빼기
        # self.image_src_processor.size['shortest_edge'] = 512
        # self.image_src_processor.crop_size = {"height": 512, "width": 512}

        self.random_erase = transforms.RandomErasing()
        self.ColorJitter_functions = {0: transforms.functional.adjust_brightness,
                                      1: transforms.functional.adjust_contrast,
                                      2: transforms.functional.adjust_saturation,
                                      3: transforms.functional.adjust_hue}

        self.transform_totensor = transforms.ToTensor()
        self.transform_normalize = transforms.Normalize([0.5], [0.5])

        # self.kpt_param = {}
        # self.kpt_param['offset'] = 40
        # self.kpt_param['stickwidth'] = 4
        # self.kpt_param['anno_width'] = 176
        # self.kpt_param['anno_height'] = 256
        # self.PK = ProcessingKeypoints()

    def __len__(self):
        return len(self.data)

    def transforms(self, source_img, target_img, s_keypoint, t_keypoint):
        # Random crop
        # 예외, text, 너무 작은
        t_kpt_pad = None
        if random.random() < 0.6:  # 0.5
            if random.random() < 0.5:  # 0.5
                crop = transforms.RandomResizedCrop(self.model_img_size)
                # _sclae = np.random.choice([0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
                params = crop.get_params(source_img, scale=(0.7, 1), ratio=(1.0, 1.0))
                source_img = transforms.functional.crop(source_img, *params)
                source_img = transforms.functional.resize(source_img, crop.size[::-1])
            else:
                if np.isin(-1, s_keypoint):
                    # s_pad_val = np.random.choice(int(self.model_img_size[0] * 0.4 // 2))
                    _sclae = np.random.choice([0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1])
                    s_pad_val = int(self.model_img_size[0] * _sclae // 2)
                    source_img = transforms.Pad(s_pad_val, fill=0, padding_mode="constant")(source_img)
                    source_img = transforms.functional.resize(source_img, self.model_img_size[::-1])

        if random.random() < 0.6:
            if random.random() < 0.5:  # 0.5
                crop = transforms.RandomResizedCrop(self.model_img_size)
                params = crop.get_params(source_img, scale=(0.7, 1), ratio=(1.0, 1.0))
                target_img = transforms.functional.crop(target_img, *params)
                target_img = transforms.functional.resize(target_img, crop.size[::-1])
                kpt_mask = t_keypoint == -1
                t_keypoint = self.kpt_cropresize(t_keypoint, params)
                t_keypoint[kpt_mask] = -1
            else:
                t_keypoint = np.concatenate((t_keypoint, [[0,0], self.model_img_size])) # pad 위치를 알기 위하여
                if np.isin(-1, t_keypoint):  # 전체 형상일 경우 augmentation하지 않음  # 0.5
                    # t_pad_val = np.random.choice(int(self.model_img_size[0] * 0.4 // 2))
                    _sclae = np.random.choice([0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1])
                    t_pad_val = int(self.model_img_size[0] * _sclae // 2)
                    target_img = transforms.Pad(t_pad_val, fill=0, padding_mode="constant")(target_img)
                    target_img = transforms.functional.resize(target_img, self.model_img_size[::-1])
                    kpt_mask = t_keypoint == -1
                    _kpt = t_keypoint + t_pad_val
                    _kpt = self.kpt_resize(_kpt, t_pad_val, self.model_img_size)
                    _kpt[kpt_mask] = -1
                    t_keypoint = _kpt[:-2,:]
                    t_kpt_pad = _kpt[-2:,:]

        if random.random() < 0.5:
            source_img = transforms.functional.hflip(source_img)
            target_img = transforms.functional.hflip(target_img)
            pt_mask = t_keypoint == -1
            t_keypoint[:, 0] = self.model_img_size[0] - t_keypoint[:, 0]
            t_keypoint[pt_mask] = -1

        # if random.random() < 0.0:  # 0.8
        #     jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        #     params = jitter.get_params(jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue)
        #
        #     for i in np.array(params[0]):
        #         source_img = self.ColorJitter_functions[i](source_img, params[i + 1])
        #         target_img = self.ColorJitter_functions[i](target_img, params[i + 1])
        #
        # if random.random() < 0.0:  # 0.2
        #     target_img = transforms.functional.rgb_to_grayscale(target_img, num_output_channels=3)
        #     source_img = transforms.functional.rgb_to_grayscale(source_img, num_output_channels=3)

        return source_img, target_img, t_keypoint, t_kpt_pad

    def __getitem__(self, idx):
        item = self.data[idx]

        s_img_path = os.path.join(self.image_root_path, item["source_image"])
        s_img = Image.open(s_img_path).resize(self.img_size, Image.BICUBIC)

        t_img_path = os.path.join(self.image_root_path, item["target_image"])
        t_img = Image.open(t_img_path).resize(self.img_size, Image.BICUBIC)

        t_pose = Image.open(t_img_path.replace("/img/", "/pose_img/")).resize(self.img_size, Image.BICUBIC)

        s_img = s_img.resize(self.model_img_size, Image.BICUBIC)
        t_img = t_img.resize(self.model_img_size, Image.BICUBIC)
        t_pose = t_pose.resize(self.model_img_size, Image.BICUBIC)

        # t_pose_path = t_img_path.replace('img', 'pose_img').replace('.jpg', '.txt')
        # t_keypoint = np.loadtxt(t_pose_path)
        # t_keypoint = self.PK.trans_keypoins(t_keypoint, self.model_img_size[::-1], self.kpt_param)
        # s_pose_path = s_img_path.replace('img', 'pose').replace('.jpg', '.txt')
        # s_keypoint = np.loadtxt(s_pose_path)
        # s_keypoint = self.PK.trans_keypoins(s_keypoint, self.model_img_size[::-1], self.kpt_param)
        # t_kpt_pad = None
        # if self.phase == 'train':
        #     s_img, t_img, t_keypoint, t_kpt_pad = self.transforms(s_img, t_img, s_keypoint, t_keypoint)
        # t_pose = self.PK.draw_img(t_keypoint, self.model_img_size[::-1], self.kpt_param)
        # if t_kpt_pad is not None:
        #     t_pose = self.add_kptpad(t_pose, t_kpt_pad)

        processed_s_img = (self.image_processor(images=s_img,
                                                return_tensors="pt").pixel_values).squeeze(dim=0)
        processed_t_pose = (self.image_processor(images=t_pose,
                                                 return_tensors="pt").pixel_values).squeeze(dim=0)

        trans_t_img = self.transform_normalize(self.transform_totensor(t_img))
        trans_t_pose = self.transform_totensor(t_pose)

        # if random.random() < self.pose_erase_rate:
        #     params = self.random_erase.get_params(trans_t_pose, scale=(0.05, 0.5), ratio=(0.3, 3.3))
        #     i, j, h, w, _ = params
        #     trans_t_pose[:, i:i + h, j:j + w] = 0

        if random.random() < self.pose_drop_rate:
            processed_t_pose = torch.zeros(processed_t_pose.shape)

        return {
            "processed_s_img": processed_s_img,
            "processed_t_pose": processed_t_pose,
            "trans_t_img": trans_t_img,
            "trans_t_pose": trans_t_pose,
        }

    def kpt_resize(self, keypoint, pad_val, resize):
        scale_w = 1.0 / (self.model_img_size[0] + pad_val * 2) * resize[0]
        scale_h = 1.0 / (self.model_img_size[1] + pad_val * 2) * resize[1]
        keypoint[:, 0] = keypoint[:, 0] * scale_w
        keypoint[:, 1] = keypoint[:, 1] * scale_h
        # pad_vals = {'x1': pad_val * scale_w, 'y1': pad_val * scale_h}
        return keypoint  # , pad_vals

    def kpt_cropresize(self, keypoint, params):
        _t_kpt_h = keypoint[:, 1] - params[0]
        _t_kpt_w = keypoint[:, 0] - params[1]
        _t_kpt_h[_t_kpt_h > params[2]] = -1
        _t_kpt_w[_t_kpt_w > params[3]] = -1
        scale_w = 1.0 / params[3] * self.model_img_size[0]
        scale_h = 1.0 / params[2] * self.model_img_size[1]
        keypoint[:, 0] = _t_kpt_w * scale_w
        keypoint[:, 1] = _t_kpt_h * scale_h
        return keypoint

    def add_kptpad(self, t_pose, t_kpt_pad):
        t_kpt_pad = t_kpt_pad.astype(int)
        t_pose = np.array(t_pose)
        t_pose[:t_kpt_pad[0][0], :, :] = 255
        t_pose[t_kpt_pad[1][0]:, :, :] = 255
        t_pose[:, :t_kpt_pad[0][1], :] = 255
        t_pose[:, t_kpt_pad[1][1]:, :] = 255
        t_pose = Image.fromarray(t_pose)
        return t_pose