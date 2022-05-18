import torch
import timm
from torch import nn


class Head(nn.Module):
    def __init__(self,
                 in_channels: int = 512,
                 feature_size: int = 512):
        super().__init__()
        self._dropout = nn.Dropout()
        self._linear = nn.Linear(in_channels, feature_size)
        self._bn1d = nn.BatchNorm1d(feature_size)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, val=1)
                nn.init.constant_(layer.bias, val=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._dropout(x)
        x = self._linear(x)
        x = self._bn1d(x)
        return x


class ArcFace(nn.Module):
    def __init__(self, backbone_name: str, out_features: int = 512):
        super().__init__()
        self.model = timm.create_model(backbone_name, pretrained=True)
        weight = self.model.conv1.weight.data.mean(1)[:, None]
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model.conv1.data = weight

        self.pool = nn.AdaptiveMaxPool2d(1)
        try:
            self.features = self.model.feature_info[-1]["num_chs"]
        except AttributeError:
            with torch.no_grad():
                x = self.model.forward_features(torch.randn(2, 1, 256, 256))
                self.features = x.shape[1]

        self.out_features = out_features
        self.head = Head(self.features, out_features)

    def forward(self, x):
        feature = self.model.forward_features(x)
        feature = self.pool(feature)
        feature = torch.flatten(feature, 1)
        return self.head(feature)
