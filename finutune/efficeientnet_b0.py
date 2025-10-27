import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision import models




def setup_efficientnet_b0(num_classes=100, pretrained=True):

    # Load pre-trained model
    model = models.efficientnet_b0(pretrained=pretrained)

    if pretrained:
        print("Loaded pre-trained weights from ImageNet")

    # Print original architecture
    print(f"Original classifier: {model.classifier}")

    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, num_classes)
    )

    print(f"Modified classifier: {model.classifier}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model