from dataclasses import dataclass
import os
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.nn import CrossEntropyLoss
from torch.multiprocessing import set_start_method
from torchvision.transforms import ToTensor, Compose, Resize

from models import LeNet5
from common import DistributedTrainer, DistributedTester
from kfold import get_scheduler

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    batch_size = 32
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5().to(device)

    optimizer = Adam(model.parameters(), 1e-3)
    scheduler = get_scheduler('cosine', optimizer)
    transforms = Compose([Resize((32, 32)), ToTensor()])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transforms)
    test_dataset = MNIST(root="./data", train=True, download=True, transform=transforms)

    trainer = DistributedTrainer(5, CrossEntropyLoss(), root_dir, batch_size)
    trainer(model, train_dataset, optimizer, scheduler, epochs=10)

    tester = DistributedTester(DataLoader(test_dataset, batch_size=batch_size, shuffle=True), root_dir)
    tester(model)
