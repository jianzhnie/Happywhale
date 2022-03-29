'''
Author: jianzhnie
Date: 2022-03-29 11:25:36
LastEditTime: 2022-03-29 11:26:43
LastEditors: jianzhnie
Description:

'''

import timm
import torch
import torch.nn as nn
from timm.optim import create_optimizer_v2

import pytorch_lightning as pl

from ..losses.focalloss import FocalLoss
from .layers.arcmargin import ArcMarginProduct


class LitModule(pl.LightningModule):
    def __init__(self, model_name, pretrained, drop_rate, embedding_size,
                 num_classes, arc_s, arc_m, arc_easy_margin, arc_ls_eps,
                 optimizer, learning_rate, weight_decay, len_train_dl, epochs):
        super().__init__()

        self.save_hyperparameters()

        self.model = timm.create_model(model_name,
                                       pretrained=pretrained,
                                       drop_rate=drop_rate)
        self.embedding = nn.Linear(self.model.get_classifier().in_features,
                                   embedding_size)
        self.model.reset_classifier(num_classes=0, global_pool='avg')

        self.arc = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=arc_s,
            m=arc_m,
            easy_margin=arc_easy_margin,
            ls_eps=arc_ls_eps,
        )

        #         self.loss_fn = F.cross_entropy
        self.loss_fn = FocalLoss()

    def forward(self, images):
        features = self.model(images)
        embeddings = self.embedding(features)

        return embeddings

    def configure_optimizers(self):
        # 优化器
        optimizer = create_optimizer_v2(
            self.parameters(),
            opt=self.hparams.optimizer,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # 学习率调整
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.hparams.learning_rate,
            steps_per_epoch=self.hparams.len_train_dl,
            epochs=self.hparams.epochs,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step'}

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def _step(self, batch, step):
        images, targets = batch['image'], batch['target']

        embeddings = self(images)
        outputs = self.arc(embeddings, targets, self.device)

        loss = self.loss_fn(outputs, targets)
        # 标记该loss，用于保存模型时监控该量
        self.log(f'{step}_loss', loss)

        return loss
