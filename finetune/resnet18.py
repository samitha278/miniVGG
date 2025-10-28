import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision import models


class ResNet18Wrapper(nn.Module):
    
    
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        
        # Replace final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x, targets=None):
        logits = self.model(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        
        return logits


def setup_resnet18(num_classes=100, pretrained=True):
    
    # Create wrapped model
    model = ResNet18Wrapper(num_classes=num_classes, pretrained=pretrained)
    
    if pretrained:
        print("Loaded pre-trained weights from ImageNet")
    
    print(f"Original final layer: Linear(in_features=512, out_features=1000, bias=True)")
    print(f"Modified final layer: {model.model.fc}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model