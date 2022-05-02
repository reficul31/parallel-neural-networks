import torch
import torch.nn.functional as F

from torch.nn import Module, Conv2d, ReLU, MaxPool2d, Sequential, Linear, BatchNorm2d

class LeNet5(Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.layer1 = Sequential(
            Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            BatchNorm2d(6),
            ReLU(),
            MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = Sequential(
            Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = Linear(400, 120)
        self.relu = ReLU()
        self.fc1 = Linear(120, 84)
        self.relu1 = ReLU()
        self.fc2 = Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out