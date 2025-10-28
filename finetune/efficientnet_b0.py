import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision import models


class EfficientNetB0Wrapper(nn.Module):
    
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=pretrained)
        
        # Replace classifier
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x, targets=None):
        logits = self.model(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        
        return logits


def setup_efficientnet_b0(num_classes=100, pretrained=True):

    # Create wrapped model
    model = EfficientNetB0Wrapper(num_classes=num_classes, pretrained=pretrained)

    if pretrained:
        print("Loaded pre-trained weights from ImageNet")

    print(f"Original classifier: Sequential(Dropout(p=0.2, inplace=True), Linear(in_features=1280, out_features=1000, bias=True))")
    print(f"Modified classifier: {model.model.classifier}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model