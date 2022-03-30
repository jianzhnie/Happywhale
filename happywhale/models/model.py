'''
Author: jianzhnie
Date: 2022-03-29 11:11:54
LastEditTime: 2022-03-29 11:47:53
LastEditors: jianzhnie
Description:

'''
import timm
import torch.nn as nn

from .layers.arcmargin import ArcMarginProduct
from .layers.gem import GeM


class HappyWhaleModel(nn.Module):
    def __init__(self,
                 model_name,
                 embedding_size=512,
                 num_classes=15587,
                 s=30.0,
                 m=0.5,
                 ls_eps=0.0,
                 easy_margin=False,
                 pretrained=True):
        super(HappyWhaleModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = ArcMarginProduct(embedding_size, num_classes, s, m,
                                   easy_margin, ls_eps)

    def forward(self, images, labels):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        output = self.fc(embedding, labels)
        return output

    def extract(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        return embedding
