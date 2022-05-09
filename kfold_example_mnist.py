import os

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torch.multiprocessing import set_start_method
from torchvision.transforms import ToTensor, Resize, Compose

from kfold import AsyncKFold
from models import LeNet5, AlexNet

batch_size = 32
def get_optimizer(model):
    return Adam(model.parameters(), 1e-3)

def get_models():
    return [LeNet5, AlexNet]

def get_schedulers():
    return ['cosine']

def get_dataset():
    transforms = Compose([Resize((32, 32)), ToTensor()])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transforms)
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
