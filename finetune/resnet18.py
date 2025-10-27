import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision import models





def setup_resnet18(num_classes=100, pretrained=True):
    
    # Load pre-trained model
    model = models.resnet18(pretrained=pretrained)
    
    if pretrained:
        print("Loaded pre-trained weights from ImageNet")
    
    # Print original architecture
    print(f"Original final layer: {model.fc}")
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    print(f"Modified final layer: {model.fc}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model