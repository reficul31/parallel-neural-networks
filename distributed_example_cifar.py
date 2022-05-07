from dataclasses import dataclass
import os
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.nn import CrossEntropyLoss
from torch.multiprocessing import set_start_method
from torchvision.transforms import ToTensor, Compose, RandomCrop, RandomHorizontalFlip, Normalize

from models import VGG
from common import DistributedTrainer, DistributedTester
from kfold import get_scheduler

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    batch_size = 32
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG().to(device)

    optimizer = Adam(model.parameters(), 1e-3)
    scheduler = get_scheduler('cosine', optimizer)
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

    trainer = DistributedTrainer(5, CrossEntropyLoss(), root_dir, batch_size)
    trainer(model, train_dataset, optimizer, scheduler, epochs=10)

    tester = DistributedTester(DataLoader(test_dataset, batch_size=batch_size, shuffle=True), root_dir)
    tester(model)
