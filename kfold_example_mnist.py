import os

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torch.multiprocessing import set_start_method
from torchvision.transforms import ToTensor, Resize, Compose

from kfold import KFold
from models import LeNet5, AlexNet

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    batch_size = 32
    root_dir = os.path.dirname(os.path.abspath(__file__))

    def get_optimizer(model):
        return Adam(model.parameters(), 1e-3)

    transforms = Compose([Resize((32, 32)), ToTensor()])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transforms)

    trainer_params = dict({"epochs": 2})
    kfold = KFold(2, train_dataset, root_dir, CrossEntropyLoss(), batch_size)
    kfold([LeNet5, AlexNet], ['cosine'], get_optimizer, trainer_params)
