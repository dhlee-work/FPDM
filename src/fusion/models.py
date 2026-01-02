import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from pytorch_metric_learning import losses
# from pytorch_metric_learning.distances import CosineSimilarity
from transformers import CLIPVisionModelWithProjection
from transformers import AutoImageProcessor, AutoModel
from torch.optim.lr_scheduler import LRScheduler
import math

class Combiner(nn.Module):
    """
    reference : https://github.com/ABaldrati/CLIP4Cir/blob/master/src/combiner.py
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, img_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        self.kpt_projection_layer = nn.Linear(img_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(img_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, img_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim),
                                            nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.logit_scale = 100

    # def forward(self, image_features: torch.tensor, kpt_features: torch.tensor,
    #             target_features: torch.tensor) -> torch.tensor:
    def forward(self, image_features: torch.tensor, kpt_features: torch.tensor) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features.
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :param target_features: CLIP target image features
        :return: scaled logits
        """
        predicted_features = F.normalize(self.combine_features(image_features, kpt_features), dim=-1)
        # target_features = F.normalize(target_features, dim=-1)
        # logits = self.logit_scale * predicted_features @ target_features.T
        # return logits
        return predicted_features

    def combine_features(self, image_features: torch.tensor, kpt_features: torch.tensor) -> torch.tensor:
        """
        Combine the reference image features and the caption features. It outputs the predicted features
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: predicted features
        """
        kpt_projected_features = self.dropout1(F.relu(self.kpt_projection_layer(kpt_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((kpt_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * kpt_features + (
                1 - dynamic_scalar) * image_features
        # return F.normalize(output, dim=-1)
        return output

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class FusionModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.encoder_type = self.hparams.encoder_type
        self.lambda_l1 = self.hparams.lambda_l1
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"

        if self.hparams.encoder_type == 'clip':
            self.img_encoder = CLIPVisionModelWithProjection.from_pretrained(self.hparams.img_encoder_path)
        elif self.hparams.encoder_type == 'dino':
            self.img_encoder = AutoModel.from_pretrained(self.hparams.img_encoder_path)
        elif self.hparams.encoder_type == 'swin':
            self.img_encoder = AutoModel.from_pretrained(self.hparams.img_encoder_path)
        else:
            assert('did not assigned proper  image encoder!! ')
        if not self.hparams.img_encoder_update:
            self.img_encoder.requires_grad_(False)
        # self.img_encoder.requires_grad_(False)

        self.pose_encoder = self.img_encoder
        self.combiner = Combiner(img_feature_dim=self.hparams.combiner_hidden_dim, projection_dim=768,
                                 hidden_dim=768)  # 768

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return [optimizer]

    def combiner_fusion_info_nce_loss(self, img_s_feats, img_t_feats, fushion_feats, mode="train"):
        # Calculate cosine similarity

        # if conbiner src_f_laerning
        if self.hparams.conbiner_self_learning:
            img_t_feats = torch.concat((img_t_feats, img_s_feats), axis=0)
            fushion_feats = torch.concat((fushion_feats, img_s_feats), axis=0)
            img_s_feats = torch.concat((img_s_feats, img_s_feats), axis=0)

        cos_sim = F.cosine_similarity(img_t_feats[:, None, :], fushion_feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        pos_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)

        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)

        l1 = torch.mean(torch.abs(img_s_feats - fushion_feats))
        nll = nll.mean() + self.lambda_l1 * l1

        # Logging loss
        self.log(mode + "_loss_fushion_l1", l1)
        self.log(mode + "_loss_fushion", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        # Logging ranking metrics
        self.log(mode + "_fushion_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_fushion_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_fushion_acc_mean_pos", 1 + sim_argsort.float().mean())
        return nll

    def training_step(self, batch, batch_idx):
        source_imgs = batch['source_img']
        target_imgs = batch['target_img']
        target_pose = batch['target_pose']

        encoded_source = self.img_encoder(source_imgs)
        encoded_target_pose = self.pose_encoder(target_pose)
        encoded_target = self.img_encoder(target_imgs)

        if self.hparams.encoder_type == 'clip':
            source_img_feats = encoded_source.image_embeds
            targets_img_feats = encoded_target.image_embeds
            target_pose_feats = encoded_target_pose.image_embeds
        else:
            source_img_feats = encoded_source.pooler_output
            targets_img_feats = encoded_target.pooler_output
            target_pose_feats = encoded_target_pose.pooler_output

        fusion_img_feats = self.combiner(source_img_feats, target_pose_feats)
        global_info_ncs_loss = self.combiner_fusion_info_nce_loss(source_img_feats,
                                                                  targets_img_feats,
                                                                  fusion_img_feats, mode="train")
        total_loss = global_info_ncs_loss
        self.log('train' + "_loss", total_loss.float().mean())
        return total_loss

    def validation_step(self, batch, batch_idx):
        source_imgs = batch['source_img']
        target_imgs = batch['target_img']
        target_pose = batch['target_pose']

        encoded_source = self.img_encoder(source_imgs)
        encoded_target = self.img_encoder(target_imgs)
        encoded_target_pose = self.img_encoder(target_pose)

        if self.hparams.encoder_type == 'clip':
            source_img_feats = encoded_source.image_embeds
            targets_img_feats = encoded_target.image_embeds
            target_pose_feats = encoded_target_pose.image_embeds
        else:
            source_img_feats = encoded_source.pooler_output
            targets_img_feats = encoded_target.pooler_output
            target_pose_feats = encoded_target_pose.pooler_output

        fusion_img_feats = self.combiner(source_img_feats, target_pose_feats)
        global_info_ncs_loss = self.combiner_fusion_info_nce_loss(source_img_feats,
                                                                  targets_img_feats,
                                                                  fusion_img_feats, mode="val")
        total_loss = global_info_ncs_loss
        self.log('val' + "_loss", total_loss.float().mean())
        return total_loss
