import argparse
import json
import os
from datetime import datetime
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import numpy as np

from src.fusion.dataset import FusionDataset
from src.fusion.models import FusionModel
from pytorch_lightning.strategies import DDPStrategy

def read_dataset(root_path, filename):
    with open(os.path.join(root_path, filename), 'r') as f:
        data = json.load(f)
    return data


def load_logger(args):
    time_now = 'model0' # datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_dir = f"logs/{args.project_name}/{time_now}/"
    os.makedirs(log_dir, exist_ok=True)
    if not args.disable_logger:
        if not args.trained_model_name:
            logger = WandbLogger(name=args.project_name,
                                 project=args.project_name,
                                 log_model=True,
                                 save_dir=log_dir)
        else:
            logger = WandbLogger(name=args.project_name,
                                 project=args.project_name,
                                 id=args.wandb_id,
                                 resume='allow',
                                 log_model=True,
                                 save_dir=log_dir)
    else:
        logger = None

    ckpt_cb = ModelCheckpoint(dirpath=log_dir,
                              monitor='val_loss',
                              mode="min",
                              every_n_epochs=1,
                              save_top_k=1,
                              save_last=True)

    return logger, ckpt_cb

def split_trainset(dataset, thr_ratio):
    thr = int(len(dataset)*thr_ratio)
    idx = np.arange(len(dataset))
    np.random.seed(7)
    np.random.shuffle(idx)
    train_dataset = list(np.array(dataset)[idx][:thr])
    valid_dataset = list(np.array(dataset)[idx][thr:])
    return train_dataset, valid_dataset

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./config/b1.fpdm-deepfashion.yaml', help='Path to config file')
    return parser.parse_args()

args = get_parser()
config = OmegaConf.load(args.config)

logger, ckpt_cb = load_logger(config)
train_dataset = read_dataset(config.root_path, config.train_dataset_name)
if  config.thr_ratio == 1:
    valid_dataset = read_dataset(config.root_path, config.test_dataset_name)
else:
    train_dataset, valid_dataset = split_trainset(train_dataset, thr_ratio=config.thr_ratio)


config.phase = 'train'
train_dataset = FusionDataset(train_dataset, config)
train_dataloader = DataLoader(train_dataset,
                              num_workers=config.num_workers,
                              batch_size=config.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

config.phase = 'train'
valid_dataset = FusionDataset(valid_dataset, config)
test_dataloader = DataLoader(valid_dataset,
                             num_workers=config.num_workers,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True)

lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    accelerator="gpu",
    devices=config.device,
    accumulate_grad_batches=config.accumulate_grad_batches,
    max_epochs=config.max_epochs,
    callbacks=[lr_monitor_cb, ckpt_cb],
    logger=logger, precision="16-mixed"
)

# Setting the seed
pl.seed_everything(7)
if config.trained_model_name:
    ckpt_path = f'./logs/{config.project_name}/{config.trained_model_name}/last.ckpt'
    model = FusionModel(config)
    trainer.fit(model, train_dataloader, test_dataloader, ckpt_path=ckpt_path)
else:
    model = FusionModel(config)
    trainer.fit(model, train_dataloader, test_dataloader)
