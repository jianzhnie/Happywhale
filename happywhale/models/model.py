'''
Author: jianzhnie
Date: 2022-03-29 11:11:54
LastEditTime: 2022-03-29 11:47:53
LastEditors: jianzhnie
Description:

'''
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from timm.optim import create_optimizer_v2

from .layers.arcmargin import ArcMarginProduct
from .layers.gem import GeM


class HappyWhaleModel(pl.LightningModule):
    def __init__(self,
                 model_name,
                 embedding_size=512,
                 num_classes=15587,
                 arc_s=30.0,
                 arc_m=0.5,
                 arc_easy_margin=False,
                 arc_ls_eps=0.0,
                 pretrained=True):
        super(HappyWhaleModel, self).__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(model_name, pretrained=pretrained)

        self.model.classifier = nn.Identity()

        self.model.global_pool = nn.Identity()

        self.pooling = GeM()

        self.embedding = nn.Linear(self.model.get_classifier().in_features,
                                   embedding_size)
        self.arc = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=arc_s,
            m=arc_m,
            easy_margin=arc_easy_margin,
            ls_eps=arc_ls_eps,
        )

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embeddings = self.embedding(pooled_features)
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
