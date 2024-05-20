import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from src.fusion.dataset import FusionDataset
from src.fusion.models import FusionModel
# param = {}
# param['scale_size'] = (256, 256)
# param['crop_param'] = (256, 256, 0, 0)
import scipy
import matplotlib.pyplot as plt

# model = KptImgSimCLR.load_from_checkpoint('./logs/kptimg-infonce-learning/2024-04-29T01-18-53/last.ckpt')
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


# batch = next(iter(train_dataloader))
def cosinesim(dat0, dat1):
    cos_list = []
    for i in range(len(dat0)):
        cos_list.append(scipy.spatial.distance.cosine(dat0[i], dat1[i]))
    return cos_list


def read_dataset(root_path, filename):
    with open(os.path.join(root_path, filename), 'r') as f:
        data = json.load(f)
    return data

from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

inputs = processor(images=image, return_tensors="pt")
inputs['pixel_values'] = torch.zeros_like(inputs['pixel_values'])
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

#
#
# def cosinesim(dat0, dat1):
#     cos_list = []
#     for i in range(len(dat0)):
#         cos_list.append(1 - scipy.spatial.distance.cosine(dat0[i], dat1[i]))
#     return cos_list

'''
contrastive learning
1. vit output of pidm
2. vit of clip4cir
3. vir out of classification
'''
# import glob
# img_list = glob.glob('./dataset/deepfashion/img/**/*.jpg', recursive=True)
# img = plt.imread(img_list[3])
# plt.imshow(img[40:54,190:204,:]);plt.show()

x = np.linspace(-0.5, 0.5, 100)
y = 1 / (1 + np.exp(50 * (0.01 - x)))
plt.plot(x, y, color='blue')
plt.vlines(0.1, 0, 1, color='black')
plt.grid()
plt.show()


def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default='deepfashion-fusion-CLIP-patch-learning',
                        help='Path to config file')
    parser.add_argument("--root_path", type=str, default='./dataset/deepfashion/', help='Path to config file')
    parser.add_argument("--phase", type=str, default='train', help='train/test')
    parser.add_argument("--disable_logger", type=str2bool, default='false')
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    return parser.parse_args()


args = get_parser()
args.batch_size = 8
args.num_workers = 8
args.combiner_hidden_dim = 768
args.lr = 1e-5
args.temperature = 0.07  # 0.07
args.weight_decay = 1e-4
args.max_epochs = 100
args.scale_size = (256, 256)
args.lambda_l1 = 0.00001
args.encoder_type = 'clip'
args.attn_hidden_dim = 1024
args.mh_attn_size = 8

args.trained_model_name = None  #
args.wandb_id = None  # '.....'
args.train_dataset_name = 'train_pairs_data.json'
args.test_dataset_name = 'test_pairs_data.json'
args.img_encoder_path = 'openai/clip-vit-large-patch14'
args.train_patch_embeddings = True
args.train_patch_embeddings_sampling_ratio = 0.05

# if args.trained_model_name:
#     run_id = args.trained_model_name.split('-')[-1]
#     wandb.init(id=run_id, resume="allow")

train_dataset = read_dataset(args.root_path, args.train_dataset_name)
test_dataset = read_dataset(args.root_path, args.test_dataset_name)

# Transformations = ContrastiveTransformations(simclr_transforms, 2)
# args.transform = simclr_transforms
train_dataset = FusionDataset(train_dataset, args)
train_dataloader = DataLoader(train_dataset,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True)

test_dataset = FusionDataset(test_dataset, args)
test_dataloader = DataLoader(test_dataset,
                             num_workers=args.num_workers,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

fusion_model = FusionModel.load_from_checkpoint(
    './logs/deepfashion-fusion-CLIP-patch-learning-wd/2024-05-17T00-20-15/last.ckpt')
fusion_model.eval()

# fusion model load
fusion_model_img_encoder = fusion_model.img_encoder
fusion_model_pose_encoder = fusion_model.pose_encoder
fusion_model_combiner = fusion_model.combiner
fusion_model_attention = fusion_model.attention
src_image_encoder = fusion_model.img_encoder

s_img_features = None
t_img_features = None
t_pose_features = None
fusion_features = None
filename_list = None

s_img_p_features = None
t_img_p_features = None
t_pose_p_features = None
fusion_p_features = None
filename_list = None

idx = 0
with torch.no_grad():
    for batch in iter(test_dataloader):
        s_image = batch['source_img'].cuda()
        t_image = batch['target_img'].cuda()
        t_pose = batch['target_pose'].cuda()

        _s_img_e = fusion_model.img_encoder(s_image)
        _t_pose_e = fusion_model.img_encoder(t_pose)
        _t_img_e = fusion_model.img_encoder(t_image)

        _s_img_f = _s_img_e.image_embeds
        _t_pose_f = _t_pose_e.image_embeds
        _t_img_f = _t_img_e.image_embeds

        _s_img_h = _s_img_e.last_hidden_state[:, 1:, :]
        _t_pose_h = _t_pose_e.last_hidden_state[:, 1:, :]
        _t_img_h = _t_img_e.last_hidden_state[:, 1:, :]

        _attn_h = fusion_model_attention(_t_pose_h, _s_img_h, _s_img_h)
        _fusion_f = fusion_model.combiner(_s_img_f, _t_pose_f)

        _fusion_f = _fusion_f.detach().cpu().numpy()
        _s_img_f = _s_img_f.detach().cpu().numpy()
        _t_pose_f = _t_pose_f.detach().cpu().numpy()
        _t_img_f = _t_img_f.detach().cpu().numpy()

        random_idx = [np.random.choice(196) for i in range(8)]

        _s_img_h = _s_img_h[:, random_idx, :].detach().cpu().numpy().reshape(-1, 768)
        _t_pose_h = _t_pose_h[:, random_idx, :].detach().cpu().numpy().reshape(-1, 768)
        _t_img_h = _t_img_h[:, random_idx, :].detach().cpu().numpy().reshape(-1, 768)
        _attn_h = _attn_h[:, random_idx, :].detach().cpu().numpy().reshape(-1, 768)

        # _path = batch['path']
        if s_img_p_features is None:
            s_img_p_features = _s_img_h
        else:
            s_img_p_features = np.concatenate((s_img_p_features, _s_img_h), axis=0)

        if t_img_p_features is None:
            t_img_p_features = _t_img_h
        else:
            t_img_p_features = np.concatenate((t_img_p_features, _t_img_h), axis=0)

        if fusion_p_features is None:
            fusion_p_features = _attn_h
        else:
            fusion_p_features = np.concatenate((fusion_p_features, _attn_h), axis=0)

        if t_pose_p_features is None:
            t_pose_p_features = _t_pose_h
        else:
            t_pose_p_features = np.concatenate((t_pose_p_features, _t_pose_h), axis=0)

        if s_img_features is None:
            s_img_features = _s_img_f
        else:
            s_img_features = np.concatenate((s_img_features, _s_img_f), axis=0)

        if t_img_features is None:
            t_img_features = _t_img_f
        else:
            t_img_features = np.concatenate((t_img_features, _t_img_f), axis=0)

        if fusion_features is None:
            fusion_features = _fusion_f
        else:
            fusion_features = np.concatenate((fusion_features, _fusion_f), axis=0)

        if t_pose_features is None:
            t_pose_features = _t_pose_f
        else:
            t_pose_features = np.concatenate((t_pose_features, _t_pose_f), axis=0)

        print(idx)
        idx += 1

idx = np.arange(len(s_img_features))
np.random.shuffle(idx)
aa = np.array(cosinesim(s_img_features, s_img_features[idx]))
plt.hist(aa, bins=15, alpha=0.8)
plt.show()
np.mean(aa)

aa = np.array(cosinesim(s_img_features, t_img_features))
bb = np.array(cosinesim(t_pose_features, t_img_features))
cc = np.array(cosinesim(fusion_features, t_img_features))
xx = np.linspace(0, 1, 100)

plt.scatter(aa, cc, alpha=0.1, c='orange')
# plt.scatter(cc, bb, alpha=0.4, c='blue')
plt.plot(xx, xx, '-', label='x=y', c='black')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

plt.hist(aa, bins=15, alpha=0.8)
plt.hist(bb, alpha=0.5)
plt.hist(cc, bins=15, alpha=0.8)
plt.show()
np.mean(aa)
np.mean(bb)
np.mean(cc)

idx = np.arange(len(s_img_p_features))
np.random.shuffle(idx)
aa = np.array(cosinesim(s_img_p_features, s_img_p_features[idx]))
plt.hist(aa, bins=15, alpha=0.8)
plt.show()
np.mean(aa)

aa = np.array(cosinesim(s_img_p_features, t_img_p_features))
bb = np.array(cosinesim(t_pose_p_features, t_img_p_features))
cc = np.array(cosinesim(fusion_p_features, t_img_p_features))
xx = np.linspace(0, 1, 100)

plt.scatter(aa, cc, alpha=0.1, c='orange')
# plt.scatter(cc, bb, alpha=0.4, c='blue')
plt.plot(xx, xx, '-', label='x=y', c='black')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

plt.hist(aa, bins=15, alpha=0.8)
plt.hist(bb, alpha=0.5)
plt.hist(cc, bins=15, alpha=0.8)
plt.show()
np.mean(aa)
np.mean(bb)
np.mean(cc)

## 집단내 거리 집단간 거리

test_dataset = read_dataset(args.root_path, args.test_dataset_name)

id_list = []
for i in test_dataset:
    id_list.append(i['source_image'].split('/')[-2])
    # id_list.append(i['target_image'].split('/')[-2])

id_list = np.array(id_list)
uni_id_list = np.unique(id_list)

sim_item = []
_item_feature = []
for i in range(len(uni_id_list)):
    _item_feature.append(np.mean(s_img_features[id_list == uni_id_list[i]], 0))
    aa = 1 - cosine_similarity(s_img_features[id_list == uni_id_list[i]])
    sim_item.append(np.mean(aa[np.triu_indices(aa.shape[0])]))
np.mean(sim_item)
np.std(sim_item)

_item_feature = np.array(_item_feature)
bb = 1 - cosine_similarity(_item_feature)
sim_within_item = bb[np.triu_indices(bb.shape[0])]
np.std(sim_within_item)

plt.hist(sim_item)
plt.hist(sim_within_item)
plt.show()

a = np.array(cosinesim(s_img_features, t_img_features))
b = np.array(cosinesim(fusion_features, t_img_features))
