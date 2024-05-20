import argparse
import json
import os
from datetime import datetime
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.fusion.dataset import FusionDataset
from src.fusion.models import FusionModel


def read_dataset(root_path, filename):
    with open(os.path.join(root_path, filename), 'r') as f:
        data = json.load(f)
    return data


def load_logger(args):
    time_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
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
                              save_top_k=1,
                              save_last=True)

    return logger, ckpt_cb


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
    parser.add_argument("--config", type=str, default='./config/b1.fpdm.yaml', help='Path to config file')
    return parser.parse_args()



args = get_parser()
config = OmegaConf.load(args.config)
# if args.trained_model_name:
#     run_id = args.trained_model_name.split('-')[-1]
#     wandb.init(id=run_id, resume="allow")

logger, ckpt_cb = load_logger(config)
train_dataset = read_dataset(config.root_path, config.train_dataset_name)
test_dataset = read_dataset(config.root_path, config.test_dataset_name)
len(train_dataset)

dat = []
for i in train_dataset:
    dat.append(os.path.split(i['source_image'])[0])
    dat.append(os.path.split(i['target_image'])[0])
dat = list(set(dat))

train_dataset = FusionDataset(train_dataset, config)
train_dataloader = DataLoader(train_dataset,
                              num_workers=config.num_workers,
                              batch_size=config.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

test_dataset = FusionDataset(test_dataset, config)
test_dataloader = DataLoader(test_dataset,
                             num_workers=config.num_workers,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True)

lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=args.max_epochs,
    callbacks=[lr_monitor_cb, ckpt_cb],
    logger=logger,
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
