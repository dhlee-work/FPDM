import argparse
import os
from datetime import datetime
from omegaconf import OmegaConf
import pytorch_lightning as pl
from diffusers.utils import check_min_version
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from src.diffusion.dataset import FPDM_Dataset, FPDM_Collate_fn
from src.diffusion.model import FPDM

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")


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
    parser.add_argument("--config", type=str, default='./config/c1.fpdm-deepfashion-dsrvd-ab6.yaml', help='Path to config file')
    return parser.parse_args()


def load_logger(config):
    time_now = 'model0' # datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_dir = f"logs/{config.project_name}/{time_now}/"
    os.makedirs(log_dir, exist_ok=True)
    if not config.disable_logger:
        if config.logger_id:
            logger = WandbLogger(name=config.project_name,
                                 project=config.project_name,
                                 log_model=True,
                                 save_dir=log_dir,
                                 id=config.logger_id,
                                 resume="allow",
                                 reinit=True
                                 )
        else:
            logger = WandbLogger(name=config.project_name,
                                 project=config.project_name,
                                 log_model=True,
                                 save_dir=log_dir,
                                 )
    else:
        logger = None

    ckpt_cb = ModelCheckpoint(dirpath=log_dir,
                              monitor='val_loss',
                              mode="min",
                              save_top_k=1,
                              save_last=True)

    return logger, ckpt_cb


args = get_parser()
config = OmegaConf.load(args.config)
pl.seed_everything(config.seed_number)
logger, ckpt_cb = load_logger(config)

traindataset = FPDM_Dataset(
    config.train_json_path,
    config.data_root_path,
    phase='train',
    model_img_size=config.model_img_size,
    input_s_img_size=config.input_s_img_size,
    input_t_img_size=config.input_t_img_size,
    read_pose_img=config.read_pose_img,
    src_encoder_path=config.src_encoder_path,
    src_encoder_size=config.src_encoder_size,
    fusion_encoder_path=config.fusion_encoder_path,
    pose_drop_rate=config.pose_drop_rate,
    pose_erase_rate=config.pose_erase_rate
)

train_dataloader = DataLoader(traindataset,
                              collate_fn=FPDM_Collate_fn,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

testdataset = FPDM_Dataset(
    config.test_json_path,
    config.data_root_path,
    phase='test',
    model_img_size=config.model_img_size,
    input_s_img_size=config.input_s_img_size,
    input_t_img_size=config.input_t_img_size,
    read_pose_img=config.read_pose_img,
    src_encoder_path=config.src_encoder_path,
    src_encoder_size=config.src_encoder_size,
    fusion_encoder_path=config.fusion_encoder_path,
    pose_drop_rate=0,
    pose_erase_rate=0
)

test_dataloader = DataLoader(
    testdataset,
    collate_fn=FPDM_Collate_fn,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=False,
    drop_last=True,
    pin_memory=True)

lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    accelerator="gpu",
    devices=config.device,
    max_epochs=config.max_epochs,
    accumulate_grad_batches=config.accumulate_grad_batches,
    callbacks=[lr_monitor_cb, ckpt_cb],
    logger=logger, precision="16-mixed")  # precision="16-mixed"

# Setting the seed
# Check whether pretrained model exists. If yes, load it and skip training
if config.trained_model_name:
    ckpt_path = f'./logs/{config.project_name}/{config.trained_model_name}/last.ckpt'
    model = FPDM(config)
    trainer.fit(model, train_dataloader, test_dataloader, model_eval_img_size=[176, 256], ckpt_path=ckpt_path)
else:
    model = FPDM(config)
    trainer.fit(model, train_dataloader,
                test_dataloader)  # , ckpt_path='./logs/deepfashion-fusion-CLIP-patch-learning-clip16/2024-05-16T13-47-50/last.ckpt'
