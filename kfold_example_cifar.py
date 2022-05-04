import os

from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torch.nn import CrossEntropyLoss
from torch.multiprocessing import set_start_method
from torchvision.transforms import ToTensor, Compose, RandomCrop, RandomHorizontalFlip, Normalize

from models import MobileNet, LeNet, VGG
from kfold import KFold

def get_optimizer(model):
    return Adam(model.parameters(), 1e-3)

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    batch_size = 32
    root_dir = os.path.dirname(os.path.abspath(__file__))

    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform_train)

    trainer_params = dict({"epochs": 10})
    kfold = KFold(3, train_dataset, root_dir, CrossEntropyLoss(), batch_size)
    kfold([MobileNet, LeNet, VGG], ['cosine', 'warm'], get_optimizer, trainer_params)
