import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from src.fusion.dataset import FusionDataset
from src.fusion.models import FusionModel
from transformers import CLIPVisionModelWithProjection
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
#
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
#
# processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
# model = AutoModel.from_pretrained('facebook/dinov2-base')
#
# inputs = processor(images=image, return_tensors="pt")
# inputs['pixel_values'] = torch.zeros_like(inputs['pixel_values'])
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state

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

# x = np.linspace(-0.5, 0.5, 100)
# y = 1 / (1 + np.exp(50 * (0.01 - x)))
# plt.plot(x, y, color='blue')
# plt.vlines(0.1, 0, 1, color='black')
# plt.grid()
# plt.show()


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
args.num_workers = 0
args.combiner_hidden_dim = 768
args.hidden_dim = 1024
args.lr = 1e-5
args.temperature = 0.07  # 0.07
args.weight_decay = 1e-4
args.max_epochs = 100
args.img_size = (176, 256)
args.scale_size = (256, 256)
args.lambda_l1 = 0.00001
args.encoder_type = 'clip'
args.attn_hidden_dim = 1024
args.mh_attn_size = 8

args.trained_model_name = None  #
args.wandb_id = None  # '.....'
args.train_dataset_name = 'train_sample_pairs_data.json'
args.test_dataset_name = 'test_pairs_data.json'
args.img_encoder_path = 'openai/clip-vit-large-patch14' # 'openai/clip-vit-base-patch16' # 'openai/clip-vit-large-patch14' # 'facebook/dinov2-base'
args.train_patch_embeddings = True
args.train_patch_embeddings_sampling_ratio = 0.05
args.img_pose_drop_rate = 0
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

args.phase = 'test'
test_dataset = FusionDataset(test_dataset, args)
test_dataloader = DataLoader(test_dataset,
                             num_workers=args.num_workers,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

fusion_model = FusionModel.load_from_checkpoint('./logs/deepfashion-fusion-CLIP-b48-p005-r03-epoch3-lr1e-5-wd1e-4/2024-06-10T05-03-40/last.ckpt')
    # './logs/deepfashion-fusion-CLIP-patch-learning-b48/2024-05-31T10-43-40/last.ckpt')
#  deepfashion-fusion-CLIP-b48-r10-p005/2024-06-07T09-32-43
# deepfashion-fusion-CLIP-patch-learning-b48-notrans-self/2024-06-02T05-22-43
# fusion_model.img_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14').to(fusion_model.device)
fusion_model.eval()



# fusion model load
# fusion_model_img_encoder = fusion_model.img_encoder
# fusion_model_pose_encoder = fusion_model.pose_encoder
fusion_model_combiner = fusion_model.combiner
fusion_model_attention = fusion_model.attention
# src_image_encoder = fusion_model.img_encoder

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

        _attn_h = fusion_model.attention(_t_pose_h, _s_img_h, _s_img_h)
        _fusion_f = fusion_model.combiner(_s_img_f, _t_pose_f)

        _fusion_f = _fusion_f.detach().cpu().numpy()
        _s_img_f = _s_img_f.detach().cpu().numpy()
        _t_pose_f = _t_pose_f.detach().cpu().numpy()
        _t_img_f = _t_img_f.detach().cpu().numpy()
        # break


        random_idx = [np.random.choice(256) for i in range(8)]

        _s_img_h = _s_img_h[:, random_idx, :].detach().cpu().numpy().reshape(-1, args.hidden_dim)
        _t_pose_h = _t_pose_h[:, random_idx, :].detach().cpu().numpy().reshape(-1, args.hidden_dim)
        _t_img_h = _t_img_h[:, random_idx, :].detach().cpu().numpy().reshape(-1, args.hidden_dim)
        _attn_h = _attn_h[:, random_idx, :].detach().cpu().numpy().reshape(-1, args.hidden_dim)

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
dd = np.array(cosinesim(s_img_features, s_img_features[idx]))

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
# plt.hist(bb, alpha=0.5)
plt.hist(cc, bins=15, alpha=0.8)
plt.hist(dd, bins=15, alpha=0.8)
plt.show()

np.mean(aa)
np.mean(bb)
np.mean(cc)
np.mean(dd)
#
#

aa = np.array(cosinesim(s_img_p_features, t_img_p_features))
bb = np.array(cosinesim(t_pose_p_features, t_img_p_features))
cc = np.array(cosinesim(fusion_p_features, t_img_p_features))
xx = np.linspace(0, 1, 100)
#
#
idx = np.arange(len(fusion_p_features))
np.random.shuffle(idx)
dd = np.array(cosinesim(fusion_p_features, fusion_p_features[idx]))
# plt.hist(dd, bins=15, alpha=0.8)
# plt.show()


plt.scatter(aa, cc, alpha=0.1, c='orange')
# plt.scatter(cc, bb, alpha=0.4, c='blue')
plt.plot(xx, xx, '-', label='x=y', c='black')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

plt.hist(aa, bins=15, alpha=0.8)
# plt.hist(bb, alpha=0.5)
plt.hist(cc, bins=15, alpha=0.8)
plt.hist(dd, bins=15, alpha=0.8)
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






from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
# import seaborn as sns
from matplotlib import pyplot as plt


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

        _attn_h = fusion_model.attention(_t_pose_h, _s_img_h, _s_img_h)
        _fusion_f = fusion_model.combiner(_s_img_f, _t_pose_f)

        _fusion_f = _fusion_f.detach().cpu().numpy()
        _s_img_f = _s_img_f.detach().cpu().numpy()
        _t_pose_f = _t_pose_f.detach().cpu().numpy()
        _t_img_f = _t_img_f.detach().cpu().numpy()
        break


_s_img_h.shape
_s_img_h = _s_img_h[:, :, :].detach().cpu().numpy()
_t_pose_h = _t_pose_h[:, :, :].detach().cpu().numpy()
_t_img_h = _t_img_h[:, :, :].detach().cpu().numpy()
_attn_h = _attn_h[:, :, :].detach().cpu().numpy()

_t_img_h[0]
_attn_h[0]

np.mean(cosinesim(_attn_h[1], _attn_h[1]))

np.mean(cosinesim(_s_img_h[2], _attn_h[2]))
np.mean(cosinesim(_t_img_h[2], _attn_h[2]))
np.mean(cosinesim(_t_img_h[2], _s_img_h[2]))


# _img = t_image[3].cpu().numpy().transpose(1,2,0)
# plt.imshow(_img)
# plt.show()

# 축소한 차원의 수를 정합니다.
n_components = 2
# TSNE 모델의 인스턴스를 만듭니다.
model = TSNE(n_components=n_components, perplexity=20)
# data를 가지고 TSNE 모델을 훈련(적용) 합니다.
X_embedded = model.fit_transform(_attn_h[2].reshape(-1, 1024)) #
# 훈련된(차원 축소된) 데이터의 첫번째 값을 출력해 봅니다.
# print(X_embedded[0])
# [65.49378 -7.3817754]

aaa = np.arange(16).reshape(4, 4)
aaa = aaa.repeat(4, 0).repeat(4, 1)
bbb = aaa.reshape(-1)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], marker='.', c=bbb, label =bbb)
# plt.legend()
plt.show()



from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=7)
kmeans.fit(_attn_h[1])
_cluster = kmeans.labels_
_cluster.reshape(16,16)
