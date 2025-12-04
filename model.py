import torch
import torch.nn as nn
import torch.nn.functional as F

class KitchenwareCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(KitchenwareCNN, self).__init__()
        
        # Input size: [100, 100, 3]
        
        # Layer 1: Conv -> BN -> ReLU
        # MATLAB: convolution2dLayer(3,8,'Padding','same')
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(8)
        
        # Layer 2: Average Pooling
        # MATLAB: averagePooling2dLayer(2,'Stride',2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Layer 3: Conv -> BN -> ReLU
        # MATLAB: convolution2dLayer(3,16,'Padding','same')
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(16)
        
        # Layer 4: Average Pooling
        # MATLAB: averagePooling2dLayer(2,'Stride',2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Layer 5: Conv -> BN -> ReLU
        # MATLAB: convolution2dLayer(3,32,'Padding','same')
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        
        # Fully Connected Layers
        # After 2 poolings (100 -> 50 -> 25), spatial size is 25x25
        self.fc1 = nn.Linear(32 * 25 * 25, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [Batch, 3, 100, 100]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Softmax is usually applied in the loss function (CrossEntropyLoss) or explicitly for inference
        return x
