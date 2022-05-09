import os

from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torch.nn import CrossEntropyLoss
from torch.multiprocessing import set_start_method
from torchvision.transforms import ToTensor, Compose, RandomCrop, RandomHorizontalFlip, Normalize

from models import MobileNet, LeNet, VGG
from kfold import AsyncKFold

batch_size = 32
def get_optimizer(model):
    return Adam(model.parameters(), 1e-3)

def get_models():
    return [MobileNet, LeNet, VGG]

def get_schedulers():
    return ['cosine', 'warm']

def get_dataset():
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    return train_dataset

def get_criterion():
    return CrossEntropyLoss()

def get_trainer_params():
    return {"epochs": 2, "save_checkpoint_frequency": 20, "print_frequency": 15}

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    root_dir = os.path.dirname(os.path.abspath(__file__))

    trainer_params = get_trainer_params()
    kfold = AsyncKFold(2, get_dataset(), root_dir, get_criterion(), batch_size)
    kfold(get_models(), get_schedulers(), get_optimizer, trainer_params)
