%%writefile model.py

import torch.nn as nn
from torchvision import models

class My_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, cfg.NUM_CLASSES)

    def forward(self, x):
        return self.model(x)
