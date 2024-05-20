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
        args.imgs_drop_rate = 0.1
        self.data = data
        self.args = args
        # self.PK_module = ProcessingKeypoints()
        self.image_processor = AutoImageProcessor.from_pretrained(self.args.img_encoder_path)  # 앞으로 빼기

        self.ColorJitter_functions = {0: transforms.functional.adjust_brightness,
                                      1: transforms.functional.adjust_contrast,
                                      2: transforms.functional.adjust_saturation,
                                      3: transforms.functional.adjust_hue}

    def __len__(self):
        return len(self.data)

    def transforms(self, source_img, target_img, pos_t_img):
        # Random crop
        crop = transforms.RandomResizedCrop(self.args.scale_size)
        params = crop.get_params(source_img, scale=(0.8, 1.0), ratio=(0.75, 1.33))
        source_img = transforms.functional.crop(source_img, *params)
        source_img = transforms.functional.resize(source_img, crop.size)

        params = crop.get_params(source_img, scale=(0.8, 1.0), ratio=(0.75, 1.33))
        target_img = transforms.functional.crop(target_img, *params)
        target_img = transforms.functional.resize(target_img, crop.size)

        pos_t_img = transforms.functional.crop(pos_t_img, *params)
        pos_t_img = transforms.functional.resize(pos_t_img, crop.size)

        # Random horizontal flipping
        if random.random() < 0.5:
            source_img = transforms.functional.hflip(source_img)
        if random.random() < 0.5:
            target_img = transforms.functional.hflip(target_img)
            pos_t_img = transforms.functional.hflip(
                pos_t_img)  # plt.imshow(pos_t_img.numpy().transpose(1,2,0)); plt.show()

        if random.random() < 0.2:  # 0.8
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
        # source_img = source_img.resize(self.args.scale_size)

        target_img = Image.open(target_img_path)
        # target_img = target_img.resize(self.args.scale_size)

        pose_path = target_img_path.replace('img', 'pose_img')
        pos_t_img = Image.open(pose_path)
        # pos_t_img = pos_t_img.resize(self.args.scale_size)


        if self.args.phase == 'train':
            source_img, target_img, pos_t_img = self.transforms(source_img, target_img, pos_t_img)


        processed_source_img = (self.image_processor(images=source_img,
                                                     return_tensors="pt").pixel_values).squeeze(dim=0)
        processed_target_img = (self.image_processor(images=target_img,
                                                     return_tensors="pt").pixel_values).squeeze(dim=0)
        processed_target_pose = (self.image_processor(images=pos_t_img,
                                                      return_tensors="pt").pixel_values).squeeze(dim=0)

        ## dropout s_img for clip
        if random.random() < self.args.imgs_drop_rate:
            processed_source_img = torch.zeros(processed_source_img.shape)
        ## dropout pos_img for clip
        if random.random() < self.args.pose_drop_rate:
            processed_target_pose = torch.zeros(processed_target_pose.shape)


        return dict(source_img=processed_source_img, target_img=processed_target_img, target_pose=processed_target_pose)


#

class SignFusionDataset(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args
        # self.PK_module = ProcessingKeypoints()
        self.image_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-large-patch14')  # 앞으로 빼기
        self.image_processor

        self.ColorJitter_functions = {0: transforms.functional.adjust_brightness,
                                      1: transforms.functional.adjust_contrast,
                                      2: transforms.functional.adjust_saturation,
                                      3: transforms.functional.adjust_hue}

    def __len__(self):
        return len(self.data)

    def transforms(self, source_img, target_img, pos_t_img):
        # Random crop
        crop = transforms.RandomResizedCrop(self.args.scale_size)
        params = crop.get_params(source_img, scale=(0.8, 1.0), ratio=(0.75, 1.33))
        source_img = transforms.functional.crop(source_img, *params)
        source_img = transforms.functional.resize(source_img, crop.size)

        params = crop.get_params(source_img, scale=(0.8, 1.0), ratio=(0.75, 1.33))
        target_img = transforms.functional.crop(target_img, *params)
        target_img = transforms.functional.resize(target_img, crop.size)

        pos_t_img = transforms.functional.crop(pos_t_img, *params)
        pos_t_img = transforms.functional.resize(pos_t_img, crop.size)

        # Random horizontal flipping
        if random.random() < 0.5:
            source_img = transforms.functional.hflip(source_img)
        if random.random() < 0.5:
            target_img = transforms.functional.hflip(target_img)
            pos_t_img = transforms.functional.hflip(
                pos_t_img)  # plt.imshow(pos_t_img.numpy().transpose(1,2,0)); plt.show()

        if random.random() < 0.8:
            jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
            params = jitter.get_params(jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue)

            for i in np.array(params[0]):
                source_img = self.ColorJitter_functions[i](source_img, params[i + 1])
                target_img = self.ColorJitter_functions[i](target_img, params[i + 1])

        if random.random() < 0.1:
            target_img = transforms.functional.rgb_to_grayscale(target_img, num_output_channels=3)
            source_img = transforms.functional.rgb_to_grayscale(source_img, num_output_channels=3)

        # target_img = transforms.functional.to_tensor(target_img)
        # src_img = transforms.functional.to_tensor(src_img)
        # src_kpt = transforms.functional.to_tensor(src_kpt)
        #
        # target_img = transforms.functional.normalize(target_img, (0.5,), (0.5,))
        # src_img = transforms.functional.normalize(src_img, (0.5,), (0.5,))
        # src_kpt = transforms.functional.normalize(src_kpt, (0.5,), (0.5,))

        return source_img, target_img, pos_t_img

    def __getitem__(self, idx):
        dat = self.data[idx]
        source_img_filename = dat['source_image']
        target_img_filename = dat['target_image']

        source_img_path = os.path.join(self.args.root_path, source_img_filename)
        target_img_path = os.path.join(self.args.root_path, target_img_filename)

        source_img = Image.open(source_img_path)
        source_img = source_img.resize(self.args.scale_size)

        target_img = Image.open(target_img_path)
        target_img = target_img.resize(self.args.scale_size)

        pose_path = target_img_path.replace('img', 'pose_img')
        pos_t_img = Image.open(pose_path)
        pos_t_img = pos_t_img.resize(self.args.scale_size)

        if self.args.phase == 'train':
            source_img, target_img, pos_t_img = self.transforms(source_img, target_img, pos_t_img)

        if self.args.encoder_type == 'clip':
            processed_source_img = (self.image_processor(images=source_img,
                                                         return_tensors="pt").pixel_values).squeeze(dim=0)
            processed_target_img = (self.image_processor(images=target_img,
                                                         return_tensors="pt").pixel_values).squeeze(dim=0)
            processed_target_pose = (self.image_processor(images=pos_t_img,
                                                          return_tensors="pt").pixel_values).squeeze(dim=0)
        else:
            source_img = transforms.functional.to_tensor(source_img)
            processed_source_img = transforms.functional.normalize(source_img, (0.5,), (0.5,))
            target_img = transforms.functional.to_tensor(target_img)
            processed_target_img = transforms.functional.normalize(target_img, (0.5,), (0.5,))
            pos_t_img = transforms.functional.to_tensor(pos_t_img)
            processed_target_pose = transforms.functional.normalize(pos_t_img, (0.5,), (0.5,))

        return dict(source_img=processed_source_img, target_img=processed_target_img, target_pose=processed_target_pose)
#
# class ImgKptDataset(Dataset):
#     def __init__(self, data, args):
#         self.data = data
#         self.args = args
#         self.transformers = args.transform
#         self.PK_module = ProcessingKeypoints()
#
#         self.param = {}
#         self.param['scale_size'] = (256, 256)
#
#     def __len__(self):
#         return len(self.data)
#
#     def readimage(self, path):
#         img_path = os.path.join(self.args.root_path, path)
#         _img = cv2.imread(img_path.replace('img', 'img_resize'))
#         _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
#         _img = cv2.resize(_img, self.param['scale_size'])
#
#         pose_path = img_path.replace('img', 'pose').replace('jpg', 'txt')
#         # pose_path = os.path.join(self.args.root_path, pose_path)
#         kpt_img, _ = self.PK_module.get_label_tensor(pose_path, _img, self.param)
#         kpt_img = kpt_img[:3,:,:]
#
#         _img = Image.fromarray(np.uint8(_img))
#         _img = self.transformers(_img)
#         return _img, kpt_img
#
#     def __getitem__(self, idx):
#         image_path = self.data[str(idx)]
#         _img, kpt_img = self.readimage(image_path)
#
#         return dict(img=_img, kpt=kpt_img, path = image_path)


# class ImgKptSimCLRDataset(Dataset):
#     def __init__(self, data, args):
#         self.data = data
#         self.args = args
#         self.transformers = args.transform
#         self.PK_module = ProcessingKeypoints()
#
#         self.param = {}
#         self.param['scale_size'] = (256, 256)
#
#     def __len__(self):
#         return len(self.data)
#
#     def transform_crop_flip(self, img, kpt_img):
#         # Random crop
#         crop = transforms.RandomResizedCrop(256)
#         params = crop.get_params(img, scale=(0.8, 1.0), ratio=(0.75, 1.33))
#         img = transforms.functional.crop(img, *params)
#         img = transforms.functional.resize(img, crop.size)
#
#         if kpt_img:
#             kpt_img = transforms.functional.crop(kpt_img, *params)
#             kpt_img = transforms.functional.resize(kpt_img, crop.size)
#         # Random horizontal flipping
#         if random.random() > 0.5:
#             img = transforms.functional.hflip(img)
#             if kpt_img:
#                 kpt_img = transforms.functional.hflip(
#                     kpt_img)  # plt.imshow(kpt_img.numpy().transpose(1,2,0)); plt.show()
#         return img, kpt_img
#
#     def __getitem__(self, idx):
#         dat = self.data[str(idx)]
#         anchor_img_path = dat['anchor']
#         pos_img_path = dat['positive']
#
#         anchor_img_path = os.path.join(self.args.root_path, anchor_img_path)
#         pos_img_path = os.path.join(self.args.root_path, pos_img_path)
#
#         anchor_img = cv2.imread(anchor_img_path)
#         anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
#         anchor_img = cv2.resize(anchor_img, self.param['scale_size'])
#
#         positive_img = cv2.imread(pos_img_path)
#         positive_img = cv2.cvtColor(positive_img, cv2.COLOR_BGR2RGB)
#         positive_img = cv2.resize(positive_img, self.param['scale_size'])
#
#         pose_path = anchor_img_path.replace('img_resize', 'pose').replace('jpg', 'txt')
#         # pose_path = os.path.join(self.args.root_path, pose_path)
#         kpt_img, _ = self.PK_module.get_label_tensor(pose_path, anchor_img, self.param)
#         kpt_img = kpt_img[:3, :, :]
#
#         anchor_img = Image.fromarray(np.uint8(anchor_img))
#         pos_img_img = Image.fromarray(np.uint8(positive_img))
#
#         anchor_img, kpt_img = self.transform_crop_flip(anchor_img, kpt_img)
#         anchor_img = self.transformers(anchor_img)
#         pos_img_img = self.transformers(pos_img_img)
#
#         return dict(anchor_img=anchor_img, positive_img=pos_img_img, kpt=kpt_img)
#
#
# class CLDataset(Dataset):
#     def __init__(self, data, args):
#         self.data = data
#         self.PK_module = ProcessingKeypoints()
#         self.args = args
#
#         self.param = {}
#         # self.param['scale_size'] = (256, 256)
#
#         self.data_idx = np.arange(len(self.data))
#
#     def __len__(self):
#         return len(self.data)
#
#     def readimage(self, img_path):
#         img_path = os.path.join(self.args.root_path, img_path)
#         _img = cv2.imread(img_path)
#         _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
#         _img = cv2.resize(_img, (256, 256))
#         return _img
#
#     def __getitem__(self, idx):
#         pair = self.data[str(idx)]
#         anchor_path = pair['anchor']
#         positive_path = pair['positive']
#         negative_path = pair['negative']
#
#         anchor_img = self.readimage(anchor_path)
#         positive_img = self.readimage(positive_path)
#         neg_img = self.readimage(negative_path)
#
#         kpt_label = 0
#         anchor_pose_path = anchor_path.replace('img_resize', 'pose').replace('jpg', 'txt')
#         anchor_pose_path = os.path.join(self.args.root_path, anchor_pose_path)
#         kpt_img, _ = self.PK_module.get_label_tensor(anchor_pose_path, anchor_img, self.param)
#
#         # p = random.uniform(0, 1)
#         # if p < 0.5:
#         #     kpt_label = 0
#         #     anchor_pose_path = anchor_path.replace('img_resize', 'pose').replace('jpg', 'txt')
#         #     anchor_pose_path = os.path.join(self.args.root_path, anchor_pose_path)
#         #     kpt_img, _ = self.PK_module.get_label_tensor(anchor_pose_path, anchor_img, self.param)
#         # else:
#         #     kpt_label = 1
#         #     new_anchor_path = anchor_path
#         #     while new_anchor_path == anchor_path:
#         #         new_idx = np.random.choice(self.data_idx)
#         #         new_pair = self.data[str(new_idx)]
#         #         new_anchor_path = new_pair['anchor']
#         #     anchor_img2 = anchor_img # self.readimage(new_anchor_path)
#         #     anchor_pose_path = new_anchor_path.replace('img_resize', 'pose').replace('jpg', 'txt')
#         #     anchor_pose_path = os.path.join(self.args.root_path, anchor_pose_path)
#         #     kpt_img, _ = self.PK_module.get_label_tensor(anchor_pose_path, anchor_img2, self.param)
#
#         kpt_img = kpt_img[:3, :, :]
#         anchor_img = Image.fromarray(np.uint8(anchor_img))
#         positive_img = Image.fromarray(np.uint8(positive_img))
#         neg_img = Image.fromarray(np.uint8(neg_img))
#         # kpt_img = Image.fromarray(np.uint8(kpt_img))
#         if self.args.transform:  # 추후 kpt 까지 augmentation 필요
#             anchor_img = self.args.transform(anchor_img)
#             positive_img = self.args.transform(positive_img)
#             neg_img = self.args.transform(neg_img)
#
#         return dict(anchor_img=anchor_img,
#                     positive_img=positive_img,
#                     neg_img=neg_img,
#                     kpt_img=kpt_img,
#                     kpt_label=kpt_label)
#
#
# class ContrastiveTransformations:
#     def __init__(self, base_transforms, n_views=2):
#         self.base_transforms = base_transforms
#         self.n_views = n_views
#
#     def __call__(self, x):
#         return [self.base_transforms(x) for i in range(self.n_views)]
#
#
# class patchSimCLRDataset(Dataset):
#     def __init__(self, data, transformers, args):
#         self.data = data
#         self.args = args
#         self.transformers = transformers
#
#         self.param = {}
#         self.param['scale_size'] = (176, 256)
#
#     def __len__(self):
#         return len(self.data)
#
#     def readimage(self, path):
#         img_path = os.path.join(self.args.root_path, path)
#         _img = cv2.imread(img_path)
#         _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
#         _img = cv2.resize(_img, self.param['scale_size'])
#
#         H, W, C = _img.shape
#         bg_offset = 25
#         patch_size = 20
#         hx = np.random.choice(np.arange(patch_size - 1, H - patch_size))
#         wx = np.random.choice(np.arange(patch_size - 1 + bg_offset, W - patch_size - bg_offset))
#         _img_patch = _img[hx:hx + patch_size, wx:wx + patch_size]
#         _img_patch = Image.fromarray(np.uint8(_img_patch))
#
#         _img_patch = self.transformers(_img_patch)
#         return _img_patch
#
#     def __getitem__(self, idx):
#         image_path = self.data[str(idx)]
#         _img = self.readimage(image_path)
#
#         # if self.args.transform is not None:
#         #     pass
#
#         return dict(img=_img)
#
#
# class SimCLRDataset(Dataset):
#     def __init__(self, data, args):
#         self.data = data
#         self.args = args
#         self.transformers = args.transform
#         self.PK_module = ProcessingKeypoints()
#
#         self.param = {}
#         self.param['scale_size'] = (256, 256)
#
#     def __len__(self):
#         return len(self.data)
#
#     def transform_crop_flip(self, img, kpt_img):
#         # Random crop
#         crop = transforms.RandomResizedCrop(256)
#         params = crop.get_params(img, scale=(0.8, 1.0), ratio=(0.75, 1.33))
#         _img_anchor = transforms.functional.crop(img, *params)
#         _img_anchor = transforms.functional.resize(_img_anchor, crop.size)
#
#         _kpt_img = transforms.functional.crop(kpt_img, *params)
#         _kpt_img = transforms.functional.resize(_kpt_img, crop.size)
#         # Random horizontal flipping
#
#         # Random crop
#         params = crop.get_params(img, scale=(0.8, 1.0), ratio=(0.75, 1.33))
#         _img_pos = transforms.functional.crop(img, *params)
#         _img_pos = transforms.functional.resize(_img_pos, crop.size)
#
#         if random.random() > 0.5:
#             _img_anchor = transforms.functional.hflip(_img_anchor)
#             _img_pos = transforms.functional.hflip(_img_pos)
#             _kpt_img = transforms.functional.hflip(_kpt_img)  # plt.imshow(kpt_img.numpy().transpose(1,2,0)); plt.show()
#         return _img_anchor, _img_pos, _kpt_img
#
#     def readimage(self, path):
#         img_path = os.path.join(self.args.root_path, path)
#         _img = cv2.imread(img_path.replace('img', 'img_resize'))
#         _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
#         _img = cv2.resize(_img, self.param['scale_size'])
#
#         pose_path = img_path.replace('img', 'pose').replace('jpg', 'txt')
#         # pose_path = os.path.join(self.args.root_path, pose_path)
#         kpt_img, _ = self.PK_module.get_label_tensor(pose_path, _img, self.param)
#         kpt_img = kpt_img[:3, :, :]
#
#         _img = Image.fromarray(np.uint8(_img))
#         _img_anchor, _img_pos, kpt_img = self.transform_crop_flip(_img, kpt_img)
#         _img_anchor = self.transformers(_img_anchor)
#         _img_pos = self.transformers(_img_pos)
#         return _img_anchor, _img_pos, kpt_img
#
#     def __getitem__(self, idx):
#         image_path = self.data[str(idx)]
#         anchor_img, pos_img_img, kpt_img = self.readimage(image_path)
#         return dict(anchor_img=anchor_img, positive_img=pos_img_img, kpt=kpt_img)
#
#
# class ImgKptSimCLRDataset(Dataset):
#     def __init__(self, data, args):
#         self.data = data
#         self.args = args
#         self.transformers = args.transform
#         self.PK_module = ProcessingKeypoints()
#
#         self.param = {}
#         self.param['scale_size'] = (256, 256)
#
#     def __len__(self):
#         return len(self.data)
#
#     def transform_crop_flip(self, anchor_img, positive_img, kpt_img):
#         # Random crop
#         crop = transforms.RandomResizedCrop(256)
#         params = crop.get_params(anchor_img, scale=(0.8, 1.0), ratio=(0.75, 1.33))
#         anchor_img = transforms.functional.crop(anchor_img, *params)
#         anchor_img = transforms.functional.resize(anchor_img, crop.size)
#
#         kpt_img = transforms.functional.crop(kpt_img, *params)
#         kpt_img = transforms.functional.resize(kpt_img, crop.size)
#
#         params = crop.get_params(anchor_img, scale=(0.8, 1.0), ratio=(0.75, 1.33))
#         positive_img = transforms.functional.crop(positive_img, *params)
#         positive_img = transforms.functional.resize(positive_img, crop.size)
#
#         # Random horizontal flipping
#         if random.random() > 0.5:
#             anchor_img = transforms.functional.hflip(anchor_img)
#             kpt_img = transforms.functional.hflip(kpt_img)  # plt.imshow(kpt_img.numpy().transpose(1,2,0)); plt.show()
#         if random.random() > 0.5:
#             positive_img = transforms.functional.hflip(positive_img)
#         return anchor_img, positive_img, kpt_img
#
#     def __getitem__(self, idx):
#         dat = self.data[str(idx)]
#         anchor_img_path = dat['anchor']
#         pos_img_path = dat['positive']
#
#         anchor_img_path = os.path.join(self.args.root_path, anchor_img_path)
#         pos_img_path = os.path.join(self.args.root_path, pos_img_path)
#
#         anchor_img = cv2.imread(anchor_img_path)
#         anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
#         anchor_img = cv2.resize(anchor_img, self.param['scale_size'])
#
#         positive_img = cv2.imread(pos_img_path)
#         positive_img = cv2.cvtColor(positive_img, cv2.COLOR_BGR2RGB)
#         positive_img = cv2.resize(positive_img, self.param['scale_size'])
#
#         pose_path = anchor_img_path.replace('img_resize', 'pose').replace('jpg', 'txt')
#         # pose_path = os.path.join(self.args.root_path, pose_path)
#         kpt_img, _ = self.PK_module.get_label_tensor(pose_path, anchor_img, self.param)
#         kpt_img = kpt_img[:3, :, :]
#
#         anchor_img = Image.fromarray(np.uint8(anchor_img))
#         pos_img_img = Image.fromarray(np.uint8(positive_img))
#
#         anchor_img, positive_img, kpt_img = self.transform_crop_flip(anchor_img, pos_img_img, kpt_img)
#         anchor_img = self.transformers(anchor_img)
#         positive_img = self.transformers(positive_img)
#
#         return dict(anchor_img=anchor_img, positive_img=positive_img, kpt=kpt_img)
#
#
# class FusionDataset(Dataset):
#     def __init__(self, data, args):
#         self.data = data
#         self.args = args
#         self.PK_module = ProcessingKeypoints()
#
#         self.param = {}
#         self.param['scale_size'] = (256, 256)
#
#         self.ColorJitter_functions = {0: transforms.functional.adjust_brightness,
#                                       1: transforms.functional.adjust_contrast,
#                                       2: transforms.functional.adjust_saturation,
#                                       3: transforms.functional.adjust_hue}
#
#     def __len__(self):
#         return len(self.data)
#
#     def transforms(self, source_img, positive_img, kpt_img):
#         # Random crop
#         crop = transforms.RandomResizedCrop(256)
#         params = crop.get_params(source_img, scale=(0.8, 1.0), ratio=(0.75, 1.33))
#         source_img = transforms.functional.crop(source_img, *params)
#         source_img = transforms.functional.resize(source_img, crop.size)
#
#         params = crop.get_params(source_img, scale=(0.8, 1.0), ratio=(0.75, 1.33))
#         positive_img = transforms.functional.crop(positive_img, *params)
#         positive_img = transforms.functional.resize(positive_img, crop.size)
#
#         kpt_img = transforms.functional.crop(kpt_img, *params)
#         kpt_img = transforms.functional.resize(kpt_img, crop.size)
#
#         # Random horizontal flipping
#         if random.random() < 0.5:
#             source_img = transforms.functional.hflip(source_img)
#         if random.random() < 0.5:
#             positive_img = transforms.functional.hflip(positive_img)
#             kpt_img = transforms.functional.hflip(kpt_img)  # plt.imshow(kpt_img.numpy().transpose(1,2,0)); plt.show()
#
#         if random.random() < 0.8:  # 0.8
#             jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
#             params = jitter.get_params(jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue)
#
#             for i in np.array(params[0]):
#                 source_img = self.ColorJitter_functions[i](source_img, params[i + 1])
#                 positive_img = self.ColorJitter_functions[i](positive_img, params[i + 1])
#
#         if random.random() < 0.1:  # 0.2
#             positive_img = transforms.functional.rgb_to_grayscale(positive_img, num_output_channels=3)
#             source_img = transforms.functional.rgb_to_grayscale(source_img, num_output_channels=3)
#
#         positive_img = transforms.functional.to_tensor(positive_img)
#         source_img = transforms.functional.to_tensor(source_img)
#
#         positive_img = transforms.functional.normalize(positive_img, (0.5,), (0.5,))
#         source_img = transforms.functional.normalize(source_img, (0.5,), (0.5,))
#
#         return source_img, positive_img, kpt_img
#
#     def transforms_test(self, source_img, positive_img, kpt_img):
#         # Random crop
#         positive_img = transforms.functional.to_tensor(positive_img)
#         source_img = transforms.functional.to_tensor(source_img)
#
#         positive_img = transforms.functional.normalize(positive_img, (0.5,), (0.5,))
#         source_img = transforms.functional.normalize(source_img, (0.5,), (0.5,))
#
#         return source_img, positive_img, kpt_img
#
#     def __getitem__(self, idx):
#         dat = self.data[str(idx)]
#         anchor_img_path = dat['anchor']
#         pos_img_path = dat['positive']
#
#         anchor_img_path = os.path.join(self.args.root_path, anchor_img_path)
#         pos_img_path = os.path.join(self.args.root_path, pos_img_path)
#
#         source_img = cv2.imread(anchor_img_path)
#         source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
#         source_img = cv2.resize(source_img, self.param['scale_size'])
#
#         positive_img = cv2.imread(pos_img_path)
#         positive_img = cv2.cvtColor(positive_img, cv2.COLOR_BGR2RGB)
#         positive_img = cv2.resize(positive_img, self.param['scale_size'])
#
#         pose_path = pos_img_path.replace('img_resize', 'pose').replace('jpg', 'txt')
#         # pose_path = os.path.join(self.args.root_path, pose_path)
#         kpt_img, _ = self.PK_module.get_label_tensor(pose_path, source_img, self.param)
#         kpt_img = kpt_img[:3, :, :]
#
#         source_img = Image.fromarray(np.uint8(source_img))
#         pos_img_img = Image.fromarray(np.uint8(positive_img))
#
#         source_img, positive_img, kpt_img = self.transforms(source_img, pos_img_img, kpt_img)
#
#         return dict(reference_img=source_img, target_img=positive_img, target_kpt=kpt_img)
