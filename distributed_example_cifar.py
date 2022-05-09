import os
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from torch.multiprocessing import set_start_method
from torchvision.transforms import ToTensor, Compose, RandomCrop, RandomHorizontalFlip, Normalize

from models import VGG
from common import DistributedTrainer, DistributedTester
from kfold import get_scheduler

epochs = 100
batch_size, num_processes = 32, 5
def get_datasets():
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset

def get_model():
    return VGG()

def get_optimizer(model):
    return Adam(model.parameters(), 1e-3)

def get_scheduler(optimizer):
    return MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model()
    model = model.to(device)

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    train_dataset, test_dataset = get_datasets()

    trainer = DistributedTrainer(num_processes, CrossEntropyLoss(), root_dir, batch_size)
    trainer(model, train_dataset, optimizer, scheduler, epochs=epochs)

    tester = DistributedTester(DataLoader(test_dataset, batch_size=batch_size, shuffle=True), root_dir)
    tester(model)
