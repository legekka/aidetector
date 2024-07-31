import torch.nn as nn
import torch
from transformers import ViTForImageClassification, ViTConfig

class ViTModel(nn.Module):
    def __init__(self, config: ViTConfig):    
        super(ViTModel, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ViTForImageClassification(
            config
        )

        if device == 'cuda':
            self.model.to(device)
            self.model.half()

    def forward(self, x):
        return self.model(x)