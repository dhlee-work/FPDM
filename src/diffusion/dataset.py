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
    src_processed_source_image = torch.stack([example["src_processed_s_img"] for example in data])
    fusion_processed_source_image = torch.stack([example["fusion_processed_s_img"] for example in data])
    fusion_processed_target_pose = torch.stack([example["fusion_processed_t_pose"] for example in data])
    target_pose = torch.stack([example["trans_t_pose"] for example in data])
    target_image = torch.stack([example["trans_t_img"] for example in data])

    source_path = [example["source_path"] for example in data]
    target_path = [example["target_path"] for example in data]
    return {
        "src_processed_source_image" : src_processed_source_image,
        "fusion_processed_source_image": fusion_processed_source_image,
        "fusion_processed_target_pose": fusion_processed_target_pose,
        "target_pose": target_pose,
        "target_image": target_image,
        'target_path': target_path,
        'source_path': source_path,
    }

class FPDM_Dataset(Dataset):
    def __init__(
            self,
            json_file,
            image_root_path,
            phase='train',
            src_encoder_path=None,
            fusion_encoder_path=None,
            read_pose_img=False,
            input_s_img_size=(352, 512),
            input_t_img_size=(352, 512),
            model_img_size=(512, 512),
            src_encoder_size=(512,512),
            imgs_drop_rate=0.0,
            pose_drop_rate=0.0,
            pose_erase_rate=0.02
    ):
        if isinstance(json_file, str):
            self.data = json.load(open(json_file))
        else:
            self.data = json_file
        self.image_root_path = image_root_path
        self.src_encoder_size = src_encoder_size
        self.phase = phase
        self.input_s_img_size = input_s_img_size
        self.input_t_img_size = input_t_img_size
        self.read_pose_img = read_pose_img
        self.model_img_size = model_img_size
        self.imgs_drop_rate = imgs_drop_rate
        self.pose_drop_rate = pose_drop_rate
        self.pose_erase_rate = pose_erase_rate
        self.src_encoder_path = src_encoder_path
        self.fusion_encoder_path = fusion_encoder_path
        self.src_image_processor = AutoImageProcessor.from_pretrained(self.src_encoder_path)  # 앞으로 빼기
        self.src_image_processor.size['shortest_edge'] = self.src_encoder_size[0]
        self.src_image_processor.do_center_crop = False

        self.fusion_image_processor = AutoImageProcessor.from_pretrained(self.fusion_encoder_path)  # 앞으로 빼기
        self.fusion_image_processor.do_center_crop = False

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
        pose_size = [375, 550]
        if random.random() < 1.0:
            kpt_shape = t_keypoint.shape
            random_noise = np.random.normal(0, 1, kpt_shape[0]*kpt_shape[1]).reshape(kpt_shape)

            pt_mask = t_keypoint == -1
            t_keypoint = t_keypoint + random_noise
            t_keypoint[pt_mask] = -1

            t_keypoint[t_keypoint < 0] = -1
            t_keypoint[:, 0][t_keypoint[:, 0] > pose_size[0]] = pose_size[0] - 1
            t_keypoint[:, 1][t_keypoint[:, 1] > pose_size[1]] = pose_size[1] - 1

        if random.random() < 0.5:
            source_img = transforms.functional.hflip(source_img)
            target_img = transforms.functional.hflip(target_img)
            pt_mask = t_keypoint == -1
            t_keypoint[:, 0] = pose_size[0] - t_keypoint[:, 0]
            t_keypoint[pt_mask] = -1
            t_keypoint = self.flip_kpt_position(t_keypoint)

        t_pose = self.PK.draw_img(t_keypoint, [pose_size[1], pose_size[0]], self.kpt_param)
        # s_pose = self.PK.draw_img(s_keypoint, [pose_size[1], pose_size[0]], self.kpt_param)
        return source_img, target_img, t_pose

    def __getitem__(self, idx):
        item = self.data[idx]

        s_img_path = os.path.join(self.image_root_path, item["source_image"])
        t_img_path = os.path.join(self.image_root_path, item["target_image"])

        if self.phase == 'train':
            if random.random() < 0.05:
                if random.random() < 0.5:
                    t_img_path = s_img_path
                else:
                    s_img_path = t_img_path
        self.input_s_img_size = (512, 512)
        if self.phase == 'train':
            if random.random() < 0.5:
                self.input_s_img_size = (256, 256)
            else:
                self.input_s_img_size = (512, 512)

        s_img = Image.open(s_img_path).resize(self.input_s_img_size, Image.BICUBIC)
        t_img = Image.open(t_img_path).resize(self.input_t_img_size, Image.BICUBIC)
        t_img = t_img.resize(self.model_img_size, Image.BICUBIC)

        if self.read_pose_img:
            t_pose = Image.open(t_img_path.replace("/img/", "/pose_img/")).convert("RGB")
            # t_pose = t_pose.resize(self.model_img_size, Image.BICUBIC)
            # s_pose = Image.open(s_img_path.replace("/img/", "/pose_img/")).convert("RGB")

        else:
            t_pose_path = t_img_path.replace('img', 'pose').replace('.jpg', '.txt')
            t_keypoint = np.loadtxt(t_pose_path)
            t_keypoint = self.PK.trans_keypoins(t_keypoint, [550, 375], self.kpt_param)
            if self.phase == 'train':
                s_img, t_img, t_pose = self.transforms(s_img, t_img, t_keypoint)
                # s_img, t_img, s_pose, t_pose = self.transforms(s_img, t_img, s_keypoint, t_keypoint)
            else:
                t_pose = self.PK.draw_img(t_keypoint, [550, 375], self.kpt_param)

        trans_t_img = self.transform_normalize(self.transform_totensor(t_img))
        trans_t_pose = self.transform_totensor(t_pose.resize(self.model_img_size, Image.BICUBIC))

        src_processed_s_img = (self.src_image_processor(images=s_img.resize(self.src_encoder_size, Image.BICUBIC), return_tensors="pt").pixel_values).squeeze(dim=0)  ###224
        fusion_processed_s_img = (self.fusion_image_processor(images=s_img.resize([224, 224], Image.BICUBIC), return_tensors="pt").pixel_values).squeeze(dim=0)
        fusion_processed_t_pose = (self.fusion_image_processor(images=t_pose.resize([224, 224], Image.BICUBIC),
                                                               return_tensors="pt").pixel_values).squeeze(dim=0)

        return {
            "src_processed_s_img": src_processed_s_img,
            "fusion_processed_s_img": fusion_processed_s_img,
            "fusion_processed_t_pose": fusion_processed_t_pose,
            "trans_t_img": trans_t_img,
            "trans_t_pose": trans_t_pose,
            "source_path" : s_img_path,
            "target_path" : t_img_path
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

    def flip_kpt_position(self, keypoints):
        new_keypoints = np.zeros_like(keypoints)

        key_flip_mapping = [[0, 0], [1, 1], [2, 5], [3, 6],
                            [4, 7], [5, 2], [6, 3], [7, 4], [8, 11],
                            [9, 12], [10, 13], [11, 8], [12, 9], [13, 10],
                            [14, 15], [15, 14], [16, 17], [17, 16]]
        for i, j in key_flip_mapping:
            new_keypoints[i] = keypoints[j]

        return new_keypoints