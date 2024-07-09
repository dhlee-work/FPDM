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

        self.set_kpt_parm()

    def __len__(self):
        return len(self.data)

    def set_kpt_parm(self):
        if 'deepfashion' in self.image_root_path:
            self.kpt_param = {}
            self.kpt_param['offset'] = 40
            self.kpt_param['stickwidth'] = 4
            self.kpt_param['anno_width'] = 176
            self.kpt_param['anno_height'] = 256
        self.PK = ProcessingKeypoints()

    def transforms(self, source_img, target_img, t_keypoint):
        # Random crop
        # t_keypoint
        if random.random() < 0.9:
            np.array(self.model_img_size) * 0.01
            kpt_shape = t_keypoint.shape
            random_noise = np.random.normal(0, 1, kpt_shape[0]*kpt_shape[1]).reshape(kpt_shape)
            t_keypoint = t_keypoint + random_noise

        if random.random() < 0.5:
            source_img = transforms.functional.hflip(source_img)
            target_img = transforms.functional.hflip(target_img)
            pt_mask = t_keypoint == -1
            t_keypoint[:, 0] = self.model_img_size[0] - t_keypoint[:, 0]
            t_keypoint[pt_mask] = -1

        return source_img, target_img, t_keypoint

    def __getitem__(self, idx):
        item = self.data[idx]

        s_img_path = os.path.join(self.image_root_path, item["source_image"])
        s_img = Image.open(s_img_path)# .resize(self.img_size, Image.BICUBIC)

        t_img_path = os.path.join(self.image_root_path, item["target_image"])
        t_img = Image.open(t_img_path)# .resize(self.img_size, Image.BICUBIC)

        # s_img = s_img.resize(self.model_img_size, Image.BICUBIC)
        t_img = t_img.resize(self.model_img_size, Image.BICUBIC)

        # t_pose = Image.open(t_img_path.replace("/img/", "/pose_img/")).resize(self.img_size, Image.BICUBIC)
        # t_pose = t_pose.resize(self.model_img_size, Image.BICUBIC)

        t_pose_path = t_img_path.replace('img', 'pose').replace('.jpg', '.txt')
        t_keypoint = np.loadtxt(t_pose_path)
        t_keypoint = self.PK.trans_keypoins(t_keypoint, self.model_img_size[::-1], self.kpt_param)

        if self.phase == 'train':
            s_img, t_img, t_keypoint = self.transforms(s_img, t_img, t_keypoint)
        t_pose = self.PK.draw_img(t_keypoint, self.model_img_size[::-1], self.kpt_param)

        processed_s_img = (self.image_processor(images=s_img,
                                                return_tensors="pt").pixel_values).squeeze(dim=0)
        processed_t_pose = (self.image_processor(images=t_pose,
                                                 return_tensors="pt").pixel_values).squeeze(dim=0)

        trans_t_img = self.transform_normalize(self.transform_totensor(t_img))
        trans_t_pose = self.transform_totensor(t_pose)


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