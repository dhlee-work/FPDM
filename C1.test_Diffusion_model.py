import argparse
import os
from datetime import datetime
import tqdm
import pytorch_lightning as pl
from diffusers.utils import check_min_version
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import numpy as np
from src.diffusion.dataset import FPDM_Dataset, FPDM_Collate_fn
from src.diffusion.model import FPDM
import matplotlib.pyplot as plt
import json

check_min_version("0.18.0.dev0")


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
    # parser.add_argument("--config", type=str, default='./config/simclr.yaml', help='Path to config file')
    parser.add_argument("--phase", type=str, default='train', help='train/test')
    parser.add_argument("--disable_logger", type=str2bool, default='false')
    parser.add_argument("--finetune_from", type=str, default='false')
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    return parser.parse_args()

args = get_parser()
args.batch_size = 10 #10
args.num_workers = 0
args.data_root_path = './dataset/deepfashion_samples'
args.train_json_path = './dataset/deepfashion_samples/train_pairs_data.json'
args.test_json_path = './dataset/deepfashion_samples/test_pairs_data.json'
args.model_img_size = (352, 512)
args.img_eval_size = (174, 256) #(256, 256)  # (352, 512)
args.input_s_img_size = (512, 512) #(256, 256) (256, 256)
args.input_t_img_size = (352, 512) # (352, 512) (174, 256)
args.src_encoder_size = (512, 512)
args.guidance_scale = 2.0
args.num_inference_steps = 20
args.seed_number = 7
args.num_images_per_prompt = 1
args.test_n_samples = 1
args.src_encoder_path= 'facebook/dinov2-large' #'facebook/dinov2-base' # 'openai/clip-vit-base-patch16'
args.src_encoder_type= 'dino' # 'clip', 'dino'
args.fusion_encoder_path= 'openai/clip-vit-large-patch14' #'facebook/dinov2-base' # 'openai/clip-vit-base-patch16'
args.fusion_encoder_type= 'clip' # 'clip', 'dino'
args.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1" # "stabilityai/stable-diffusion-2-1-base"
args.diffusion_model_path = './logs/diffusion-deepfashion-clipdino-st1ep5-b42-cum3-size352-kptaug-spose-ablation6/model0/last.ckpt'
args.save_dir_prefix = 'generate_ab6'
args.visualize_images = True
args.calculate_metrics = True
args.phase = 'test'
pl.seed_everything(args.seed_number)


testdataset = FPDM_Dataset(
    args.test_json_path,
    args.data_root_path,
    phase='test',
    read_pose_img=True,
    src_encoder_path=args.src_encoder_path,
    fusion_encoder_path=args.fusion_encoder_path,
    model_img_size=args.model_img_size,
    input_t_img_size=args.input_t_img_size,
    input_s_img_size=args.input_s_img_size,
    src_encoder_size=args.src_encoder_size,
    pose_drop_rate=0,
    pose_erase_rate=0
)


test_dataloader = DataLoader(
    testdataset,
    collate_fn=FPDM_Collate_fn,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    pin_memory=True)

model = FPDM.load_from_checkpoint(args.diffusion_model_path, src_encoder_size=args.src_encoder_size)

model.init_device()
model.eval()
model.hparams.num_images_per_prompt = 1
model.hparams.guidance_scale = 2.0
model.hparams.num_inference_steps = 20
# model.hparams.proj_drop_rate = 0
mod = 'generation'

train_json_path = args.train_json_path
test_json_path = args.test_json_path
data = json.load(open(test_json_path))
data = [[i, j]for i, j in enumerate(data)]
i=0
for i, j in enumerate([20]):
    model.hparams.num_inference_steps = j
    # model.hparams.seed_number = i
    dname = f'{args.save_dir_prefix}_{j}_{i}'
    idx = 0
    for batch in tqdm.tqdm(test_dataloader):
        g_img_paths = batch['target_path']
        s_img_paths = batch['source_path']

        exist_mask = []
        for idx, (s_img_path, g_img_path) in enumerate(zip(s_img_paths, g_img_paths)):
            dir_path, gen_path = g_img_path.split('./img')
            dir_path, src_path = s_img_path.split('./img')
            gen_path = gen_path.replace('/', '')
            src_path = src_path.replace('/', '')
            filename = src_path + '_2_' + gen_path
            dir_path = os.path.join(dir_path, dname, filename+'.png')
            if os.path.exists(dir_path):
                exist_mask.append(True)
            else:
                exist_mask.append(False)


        # delete duplicated
        if np.sum(exist_mask) == len(exist_mask):
            continue
        if np.sum(exist_mask) > 0:
            for key in batch.keys():
                if type(batch[key]) == list:
                    batch[key] = np.array(batch[key])
                batch[key] = batch[key][~np.array(exist_mask)]

        output = model.batch_image_generation(batch, mod)

        g_img_paths = batch['target_path']
        s_img_paths = batch['source_path']

        for idx, (s_img_path, g_img_path) in enumerate(zip(s_img_paths, g_img_paths)):
            # g_img_path = g_img_path.replace('img', 'generate')
            dir_path, gen_path = g_img_path.split('./img')
            dir_path, src_path = s_img_path.split('./img')
            gen_path = gen_path.replace('/', '')
            src_path = src_path.replace('/', '')
            filename = src_path + '_2_' + gen_path
            dir_path = os.path.join(dir_path, dname, filename+'.png')
            dirname = os.path.dirname(dir_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            output[idx]['generate_images'][0].save(os.path.join(dir_path), format='PNG')
            idx += 1
