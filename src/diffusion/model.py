import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from PIL import Image, ImageEnhance
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch import nn
from torchvision import transforms
from transformers import AutoImageProcessor
# from transformers import CLIPVisionModelWithProjection
from transformers import AutoModel
# from transformers import Dinov2Model
from src.diffusion.models.unet_2d_condition import UNet2DConditionModel
from src.diffusion.pipelines.pipeline import FPDM_DiffusionPipeline
from src.fusion.models import FusionModel
from src.metrics.metrics import FID, LPIPS, Reconstruction_Metrics
from transformers import CLIPVisionModelWithProjection
import math
from torch.optim.lr_scheduler import LRScheduler
import random
import matplotlib.pyplot as plt

class WarmupStepLR(LRScheduler):
    def __init__(self, optimizer, warmup_steps, step_size, gamma=0.1, base_lr=1e-3, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = base_lr
        super(WarmupStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_steps for _ in self.base_lrs]
        else:
            # StepLR phase
            steps_since_warmup = self.last_epoch - self.warmup_steps
            factor = self.gamma ** (steps_since_warmup // self.step_size)
            return [base_lr * factor for base_lr in self.base_lrs]

class SrcImage_ProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FusionPatch_ProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

# class Patch_embedding_proj(nn.Module):
#     """
#     Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
#     [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
#     training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
#     convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
#     (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
#     model) to encode image-space conditions ... into feature maps ..."
#     """
#
#     def __init__(
#         self,
#         in_dim: int = 1024,
#         hidden_dim: int = 768,
#         out_dim: int = 320,
#         dropout: float = 0.2,
#         upsmaple_size: int = 4
#     ):
#         super().__init__()
#
#         self.hidden_dim = hidden_dim
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, self.hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.LayerNorm(self.hidden_dim),
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.LayerNorm(self.hidden_dim),
#         )
#         self.upsample = nn.Upsample(scale_factor=upsmaple_size, mode='nearest')
#         self.conv_out = nn.Conv2d(self.hidden_dim, out_dim, kernel_size=1, padding=0)
#
#     def forward(self, embed):
#         embed = self.net(embed)
#         embed = embed.transpose(2, 1)
#         embed = embed.view(-1, self.hidden_dim, 16, 16)
#         embed = self.upsample(embed)
#         embed = self.conv_out(embed)
#         return embed


class SDModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, unet, args) -> None:
        super().__init__()
        self.args = args

        # if args.model_img_size[0] == 256:
        #     upsmaple_size = 2
        # elif args.model_img_size[0] == 512:
        #     upsmaple_size = 4
        # else:
        #     print('model only can compute img size 256 or 512')

        self.srcimage_proj_model = SrcImage_ProjModel(in_dim=self.args.patch_proj_in_dim, hidden_dim=768, out_dim=1024,
                                                   dropout=self.args.proj_drop_rate)

        # if self.args.fusion_patch_embed_ahead == True:
        #     self.fusionpatch_proj_model = Patch_embedding_proj(in_dim=self.args.patch_proj_in_dim, hidden_dim=768, out_dim=320,
        #                                                    dropout=self.args.proj_drop_rate, upsmaple_size=upsmaple_size)
        # else:

        self.fusionpatch_proj_model = FusionPatch_ProjModel(in_dim=self.args.patch_proj_in_dim, hidden_dim=768, out_dim=1024,
                                                   dropout=self.args.proj_drop_rate)
        self.pose_proj = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=320,
            block_out_channels=(16, 32, 96, 256),
            conditioning_channels=3)

        self.unet = unet
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def forward(self, noisy_latents, timesteps,
                cond_src_image_feature, cond_img_patch_feature,
                cond_attn_patch_feature, pose_f, phase):

        if self.args.fusion_image_patch_encoder == True:
            extra_patch_embeddings_s = self.srcimage_proj_model(cond_img_patch_feature)
            extra_patch_embeddings_f = self.fusionpatch_proj_model(cond_attn_patch_feature)
            # if self.args.proj_fusion_image_patch_encoder == True:
            #     extra_patch_embeddings_f = self.fusionpatch_proj_model(cond_attn_patch_feature)
            # else:
            #     extra_patch_embeddings_f = cond_attn_patch_feature
            # if phase == 'train':
            #     if random.random() < self.args.module_drop_rate:
            #         extra_patch_embeddings_s = torch.zeros(extra_patch_embeddings_s.shape).to(self.unet.device)
            #     if random.random() < self.args.module_drop_rate:
            #         extra_patch_embeddings_f = torch.zeros(extra_patch_embeddings_f.shape).to(self.unet.device)
            if self.args.proj_embd_concat == True:
                encoder_image_hidden_states = torch.cat((extra_patch_embeddings_s, extra_patch_embeddings_f), axis =1)
            elif self.args.proj_embd_concat == False and self.args.fusion_patch_embed_ahead == True:
                encoder_image_hidden_states = extra_patch_embeddings_s
            else:
                encoder_image_hidden_states = extra_patch_embeddings_s + extra_patch_embeddings_f
        else:
            extra_patch_embeddings_s = self.srcimage_proj_model(cond_img_patch_feature)
            if phase == 'train':
                if random.random() < self.args.module_drop_rate:
                    extra_patch_embeddings_s = torch.zeros(extra_patch_embeddings_s.shape).to(self.unet.device)
            encoder_image_hidden_states = extra_patch_embeddings_s


        if self.args.fusion_image_encoder == True:
            if phase == 'train':
                if random.random() < self.args.module_drop_rate:
                    cond_src_image_feature = torch.zeros(cond_src_image_feature.shape).to(self.unet.device)
        else:
            cond_src_image_feature = None

        if self.args.fusion_image_patch_encoder == True and self.args.fusion_patch_embed_ahead == True:
            pose_cond = extra_patch_embeddings_f
        else:
            pose_cond = self.pose_proj(pose_f)

        if self.args.pose_module_drop == True:
            pose_cond = torch.zeros(pose_cond.shape).to(self.unet.device)

        pred_noise = self.unet(noisy_latents, timesteps, class_labels=cond_src_image_feature,
                               encoder_hidden_states=encoder_image_hidden_states,
                               my_pose_cond=pose_cond).sample
        return pred_noise


class FPDM(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.save_hyperparameters(args)
        # load train and test dataset list
        if os.path.exists(self.hparams.train_json_path):
            self.train_data_list = json.load(open(self.hparams.train_json_path))
        else:
            self.train_data_list = None
        if os.path.exists(self.hparams.test_json_path):
            self.test_data_list = json.load(open(self.hparams.test_json_path))
        else:
            self.test_data_list = None

        self.save_hyperparameters(self.args)

        # Load model
        self.vae = AutoencoderKL.from_pretrained(self.hparams.pretrained_model_name_or_path, subfolder="vae")
        if self.hparams.fusion_image_encoder == True:
            self.class_embed_type = self.hparams.class_embed_type # "projection"
        else:
            self.class_embed_type = None

        self.unet = UNet2DConditionModel.from_pretrained(self.hparams.pretrained_model_name_or_path, subfolder="unet",
                                                         in_channels=4, class_embed_type=self.class_embed_type,
                                                         projection_class_embeddings_input_dim=768,
                                                         conv_in_kernel=self.hparams.unet_conv_in_kenel,
                                                         low_cpu_mem_usage=False, ignore_mismatched_sizes=True)

        # fusion model load
        self.fusion_model = FusionModel.load_from_checkpoint(self.hparams.fusion_model_path)
        self.fusion_model_img_encoder = self.fusion_model.img_encoder
        self.fusion_model_pose_encoder = self.fusion_model.pose_encoder
        self.fusion_model_combiner = self.fusion_model.combiner
        self.fusion_model_attention = self.fusion_model.attention

        if self.hparams.init_src_image_encoder:
            if self.hparams.src_encoder_type == 'clip':
                self.src_image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.hparams.src_encoder_path)
            else:
                self.src_image_encoder = AutoModel.from_pretrained(self.hparams.src_encoder_path)
        else:
            self.src_image_encoder = self.fusion_model.img_encoder

        # Load scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.hparams.pretrained_model_name_or_path,
                                                             subfolder="scheduler")  # beta_end=0.1)#

        self.src_image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.fusion_model.requires_grad_(False)
        self.fusion_model_img_encoder.requires_grad_(False)
        self.fusion_model_pose_encoder.requires_grad_(False)
        self.fusion_model_combiner.requires_grad_(False)
        self.fusion_model_attention.requires_grad_(False)

        self.sd_model = SDModel(unet=self.unet, args=self.hparams)

        self.pipe = FPDM_DiffusionPipeline.from_pretrained(self.hparams.pretrained_model_name_or_path)
        self.pipe.unet = self.unet
        # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe.scheduler = DDPMScheduler.from_pretrained(self.hparams.pretrained_model_name_or_path,
                                                            subfolder="scheduler")
        self.pipe.enable_xformers_memory_efficient_attention()

        self.image_processor = AutoImageProcessor.from_pretrained(self.hparams.src_encoder_path)  # 앞으로 빼기

        self.transform_totensor = transforms.ToTensor()
        self.transform_normalize = transforms.Normalize([0.5], [0.5])

        self.fid = FID()
        self.lpips_obj = LPIPS()
        self.rec = Reconstruction_Metrics()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        lr_scheduler = WarmupStepLR(optimizer = optimizer,
                                    warmup_steps = self.hparams.scheduler_t_up,
                                    step_size = self.hparams.scheduler_step_size,
                                    gamma = self.hparams.scheduler_gamma,
                                    base_lr=self.hparams.scheduler_eta_max)

        return [optimizer], [lr_scheduler]

    def image_grid(self, imgs, rows, cols):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        # grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    def data_load(self, s_img_path, t_img_path, t_pose_path):
        # load images
        s_img = Image.open(s_img_path).convert("RGB").resize((self.hparams.img_size),
                                                             Image.BICUBIC)
        if t_img_path:
            t_img = Image.open(t_img_path).convert("RGB").resize((self.hparams.img_size),
                                                                 Image.BICUBIC)
        else:
            t_img = None

        t_pose = Image.open(t_pose_path).convert("RGB").resize((self.hparams.img_size),
                                                               Image.BICUBIC)


        s_img = s_img.resize(self.hparams.model_img_size, Image.BICUBIC)
        t_img = t_img.resize(self.hparams.model_img_size, Image.BICUBIC)
        t_pose = t_pose.resize(self.hparams.model_img_size, Image.BICUBIC)

        # preprocessing
        trans_s_img = self.transform_totensor(s_img)
        trans_s_img = self.transform_normalize(trans_s_img).to(self.dtype).to(self.device).unsqueeze(0)
        if t_img_path:
            trans_t_img = self.transform_totensor(t_img)
            trans_t_img = self.transform_normalize(trans_t_img).to(self.dtype).to(self.device).unsqueeze(0)
        else:
            trans_t_img = None

        trans_t_pose = self.transform_totensor(t_pose).to(self.dtype).to(self.device).unsqueeze(0)

        processed_s_img = (self.image_processor(images=s_img.resize((224, 224)),
                                                return_tensors="pt").pixel_values).to(self.dtype).to(self.device)
        processed_t_pose = (self.image_processor(images=t_pose.resize((224, 224)),
                                                 return_tensors="pt").pixel_values).to(self.dtype).to(self.device)
        _dict = {"source_image": trans_s_img,
                 "target_image": trans_t_img,
                 "target_pose": trans_t_pose,
                 "processed_source_image": processed_s_img,
                 "processed_target_pose": processed_t_pose,
                 }
        return _dict

    def denoising_learning(self, target_imgs, target_pose, processed_source_image, processed_target_pose):

        with torch.no_grad():
            # Convert images to latent space
            latents = self.vae.encode(target_imgs).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            cond_src_feature = self.src_image_encoder(processed_source_image)
            s_image_patch_embeddings = (cond_src_feature.last_hidden_state)[:, 1:, :]

            # get fusion embeddings
            embddings_src = self.fusion_model_img_encoder(processed_source_image)
            embddings_t_pos = self.fusion_model_pose_encoder(processed_target_pose)

            if self.hparams.src_encoder_type == 'clip':
                s_image_embeddings = embddings_src.image_embeds
                t_pose_embeddings = embddings_t_pos.image_embeds
            else:
                s_image_embeddings = embddings_src.pooler_output
                t_pose_embeddings = embddings_t_pos.pooler_output
            kv_s_image_patch_embeddings = embddings_src.last_hidden_state[:, 1:, :]
            q_t_pose_patch_embeddings = embddings_t_pos.last_hidden_state[:, 1:, :]

            cond_fusion_image_feature = self.fusion_model_combiner(s_image_embeddings,
                                                                   t_pose_embeddings).unsqueeze(1)
            # cond_attn_patch_embeddings = s_image_patch_embeddings # 주의
            cond_attn_patch_embeddings = self.fusion_model_attention(q_t_pose_patch_embeddings,
                                                                     kv_s_image_patch_embeddings,
                                                                     kv_s_image_patch_embeddings)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.hparams.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.hparams.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (self.hparams.batch_size,),
                                  device=latents.device, )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noisy_latents = noisy_latents
        # Get the text embedding for conditioning

        # Predict the noise residual
        model_pred = self.sd_model(noisy_latents, timesteps,
                                   cond_fusion_image_feature,
                                   s_image_patch_embeddings,
                                   cond_attn_patch_embeddings,
                                   target_pose, self.hparams.phase)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        if self.hparams.loss_type == 'mse_loss':
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        elif self.hparams.loss_type == 'shrinkage_loss':
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = 1 / (1 + torch.exp(self.hparams.shrinkage_a * (self.hparams.shrinkage_c - loss)))
            loss = loss.mean()
        return loss

    def denosing_pip(self, target_imgs, target_pose,
                     processed_source_image,
                     processed_target_pose):

        generator = torch.Generator().manual_seed(self.hparams.seed_number)

        cond_src_feature = self.src_image_encoder(processed_source_image)
        s_image_patch_embeddings = (cond_src_feature.last_hidden_state)[:, 1:, :]

        # cond_src_feature = self.src_image_encoder(processed_source_image)
        # s_image_patch_embeddings = (cond_src_feature.last_hidden_state)[:, 1:, :]

        # get fusion embeddings
        embddings_src = self.fusion_model_img_encoder(processed_source_image)
        embddings_t_pos = self.fusion_model_pose_encoder(processed_target_pose)
        if self.hparams.src_encoder_type == 'clip':
            s_image_embeddings = embddings_src.image_embeds
            t_pose_embeddings = embddings_t_pos.image_embeds
        else:
            s_image_embeddings = embddings_src.pooler_output
            t_pose_embeddings = embddings_t_pos.pooler_output

        # image embeddings fusion
        if self.hparams.fusion_image_encoder:
            cond_fusion_image_feature = self.fusion_model_combiner(s_image_embeddings,
                                                                   t_pose_embeddings).unsqueeze(1)
        else:
            cond_fusion_image_feature = None
        # patch embeddings
        #### src image patch embeddings
        s_img_proj_f = self.sd_model.srcimage_proj_model(s_image_patch_embeddings)

        if self.hparams.fusion_image_patch_encoder:
            ##### pred target image patch embeddings fusion
            kv_s_image_patch_embeddings = embddings_src.last_hidden_state[:, 1:, :]
            q_t_pose_patch_embeddings = embddings_t_pos.last_hidden_state[:, 1:, :]
            cond_attn_patch_embeddings = self.fusion_model_attention(q_t_pose_patch_embeddings,
                                                                     kv_s_image_patch_embeddings,
                                                                     kv_s_image_patch_embeddings)
            cond_image_feature_f = self.sd_model.fusionpatch_proj_model(cond_attn_patch_embeddings)
            # if self.hparams.proj_fusion_image_patch_encoder:
            #     cond_image_feature_f = self.sd_model.fusionpatch_proj_model(cond_attn_patch_embeddings)
            # else:
            #     cond_image_feature_f = cond_attn_patch_embeddings
        else:
            cond_image_feature_f = None

        # pose image embedding
        # t_pose_f = self.sd_model.pose_proj(target_pose)
        if self.hparams.fusion_image_patch_encoder and self.hparams.fusion_patch_embed_ahead:
            t_pose_f = cond_image_feature_f
        else:
            t_pose_f = self.sd_model.pose_proj(target_pose)

        if self.hparams.pose_module_drop == True:
            t_pose_f = torch.zeros(t_pose_f.shape).to(self.unet.device)

        output = self.pipe(
            height=self.hparams.model_img_size[1],
            width=self.hparams.model_img_size[0],
            guidance_rescale=0.0,
            t_image=target_imgs,
            s_img_proj_f=s_img_proj_f,
            t_pose_f=t_pose_f,
            fusion_img_embed=cond_fusion_image_feature,
            pred_t_img_embed=cond_image_feature_f,
            num_images_per_prompt=self.hparams.num_images_per_prompt,
            guidance_scale=self.hparams.guidance_scale,
            generator=generator,
            num_inference_steps=self.hparams.num_inference_steps,
            args=self.hparams
        )
        return output

    def one_image_generation(self, s_img_path, t_img_path, t_pose_path, mod):
        out_dict = {}
        prep_data = self.data_load(s_img_path, t_img_path, t_pose_path)
        # source_imgs = prep_data['source_image']
        target_imgs = prep_data['target_image']
        target_pose = prep_data['target_pose']
        processed_source_image = prep_data['processed_source_image']
        processed_target_pose = prep_data['processed_target_pose']
        output = self.denosing_pip(target_imgs, target_pose,
                                   processed_source_image, processed_target_pose)

        if mod == 'generation':
            out_dict['source_image'] = Image.open(s_img_path).convert("RGB").resize(self.hparams.img_size,
                                                                                    Image.BICUBIC)
            out_dict['target_pose'] = Image.open(t_pose_path).convert("RGB").resize(self.hparams.img_size,
                                                                                    Image.BICUBIC)
            out_dict['generate_images'] = [i.resize(self.hparams.img_size, Image.BICUBIC) for i in output.images]
            return out_dict

        out_dict['source_image'] = Image.open(s_img_path).convert("RGB").resize(self.hparams.img_size,
                                                                                Image.BICUBIC)
        out_dict['target_image'] = Image.open(t_img_path).convert("RGB").resize(self.hparams.img_size,
                                                                                Image.BICUBIC)
        out_dict['target_pose'] = Image.open(t_pose_path).convert("RGB").resize(self.hparams.img_size,
                                                                                Image.BICUBIC)
        out_dict['generate_images'] = [i.resize(self.hparams.img_size, Image.BICUBIC) for i in output.images]

        if self.hparams.calculate_metrics:
            trans_target = np.expand_dims(np.array(out_dict['target_image']).transpose(2, 0, 1), 0) / 255.
            trans_target = np.repeat(trans_target, self.hparams.num_images_per_prompt, 0)
            trans_generate = np.concatenate(
                [np.expand_dims(np.array(i).transpose(2, 0, 1), 0) / 255. for i in out_dict['generate_images']], axis=0)

            # FID
            fid_values = self.fid(trans_generate, trans_target)
            out_dict['fid'] = fid_values

            # lpips
            lpips_values = self.lpips_obj(torch.tensor((trans_generate * 2) - 1).to(self.device, torch.float),
                                          torch.tensor((trans_target * 2) - 1).to(self.device, torch.float))
            out_dict['lpips'] = float(torch.mean(lpips_values.view(-1)).detach().cpu().numpy())

            ssim_values = []
            psnr_values = []
            for gen_img in out_dict['generate_images']:
                ssim_values.append(compare_ssim(np.array(out_dict['target_image']), np.array(gen_img),
                                                gaussian_weights=True, sigma=1.2,
                                                use_sample_covariance=False, multichannel=True, channel_axis=2,
                                                data_range=np.array(gen_img).max() - np.array(gen_img).min()
                                                ))
                psnr_values.append(
                    compare_psnr(np.array(out_dict['target_image']) / 255.0, np.array(gen_img) / 255.0, data_range=1))
            mean_ssim_value = sum(ssim_values) / self.hparams.num_images_per_prompt
            mean_psnr_value = sum(psnr_values) / self.hparams.num_images_per_prompt
            out_dict['ssim'] = mean_ssim_value
            out_dict['psnr'] = mean_psnr_value
        else:
            out_dict['ssim'] = None
            out_dict['psnr'] = None
            out_dict['lpips'] = None
            out_dict['fid'] = None

        if self.hparams.visualize_images:
            _query = [out_dict['source_image'], out_dict['target_pose'], out_dict['target_image']]
            out_dict['total_images'] = _query + out_dict['generate_images']

        return out_dict

    def init_device(self):
        self.pipe.to(self.device)
        self.lpips_obj.model.to(self.device)

    def epoch_end_run(self, mod):
        # self.vae.to(torch.float16)
        # self.pipe.to(self.device)
        # self.lpips_obj.model.to(self.device)
        self.init_device()

        metric_list = ['ssim', 'psnr', 'lpips', 'fid']
        total_metric = {i: [] for i in metric_list}
        total_images = []
        n_samples = self.hparams.test_n_samples
        if mod == 'train':
            sampled_data = np.random.choice(self.train_data_list, n_samples, replace=False)
        else:
            sampled_data = np.random.choice(self.test_data_list, n_samples, replace=False)
        for i in range(n_samples):
            _dat = sampled_data[i]
            s_img_path = os.path.join(self.hparams.data_root_path, _dat['source_image'])  # png
            t_img_path = os.path.join(self.hparams.data_root_path, _dat['target_image'])  # png
            t_pose_path = os.path.join(self.hparams.data_root_path,
                                       _dat['target_image'].replace('img', 'pose_img'))  # png
            output = self.one_image_generation(s_img_path, t_img_path, t_pose_path, mod)
            for m in metric_list:
                total_metric[m].append(output[m])
            total_images += output['total_images']

        if self.hparams.calculate_metrics:
            for m in metric_list:
                _mean = sum(total_metric[m]) / len(total_metric[m])
                self.log(f'{mod}_' + f"mean_{m}_samples-{n_samples}", _mean)

        if self.hparams.visualize_images:
            gird_images = self.image_grid(total_images, n_samples, 3 + self.hparams.num_images_per_prompt)
            self.logger.log_image(key=f"{mod}_samples", images=[gird_images])

    def training_step(self, batch):
        self.hparams.phase = 'train'
        target_imgs = batch['target_image']
        target_pose = batch['target_pose']
        processed_source_image = batch['processed_source_image']
        processed_target_pose = batch['processed_target_pose']
        total_loss = self.denoising_learning(target_imgs,
                                             target_pose,
                                             processed_source_image,
                                             processed_target_pose)

        self.log('train_' + "loss", total_loss.mean())
        return total_loss

    def validation_step(self, batch):
        self.hparams.phase = 'val'
        target_imgs = batch['target_image']
        target_pose = batch['target_pose']
        processed_source_image = batch['processed_source_image']
        processed_target_pose = batch['processed_target_pose']
        total_loss = self.denoising_learning(target_imgs,
                                             target_pose,
                                             processed_source_image,
                                             processed_target_pose)

        self.log('val_' + "loss", total_loss.mean())
        return total_loss

    def on_train_epoch_end(self):
        self.hparams.phase = 'train'
        self.epoch_end_run(self.hparams.phase)

    def on_validation_epoch_end(self):
        self.hparams.phase = 'val'
        self.epoch_end_run(self.hparams.phase)

