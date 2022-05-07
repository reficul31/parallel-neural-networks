from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Linear

class AlexNet(Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = Sequential(
            Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 192, kernel_size=5, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(192, 384, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return 