import torch
from torch import nn


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 10)
    
    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.fc1(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 3 convolutional layers (32, 64, 128 filters), each followed by ReLU + MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        # After 3 pooling layers on 32x32 -> 4x4 feature maps
        # Flattened size = 128 * 4 * 4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        #raise NotImplementedError

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        #raise NotImplementedError


class ResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18, self).__init__()
        # Load resnet18 without pretrained weights to avoid state dict mismatch on save/load
        # Set pretrained=False unless you want to fine-tune from pretrained weights
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
        # Replace the final FC layer to output 10 classes (CIFAR10)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        
    def forward(self, x):
        return self.resnet(x)
          



