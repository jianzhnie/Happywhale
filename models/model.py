import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.arcmargin import ArcMarginProduct
from .layers.gem import GeM
import timm


class HappyWhaleModel(nn.Module):
    def __init__(self, model_name, embedding_size, pretrained=True):
        super(HappyWhaleModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = ArcMarginProduct(embedding_size,
                                   CONFIG["num_classes"],
                                   s=CONFIG["s"],
                                   m=CONFIG["m"],
                                   easy_margin=CONFIG["ls_eps"],
                                   ls_eps=CONFIG["ls_eps"])

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