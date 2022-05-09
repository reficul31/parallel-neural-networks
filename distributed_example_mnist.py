import os
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from torch.multiprocessing import set_start_method
from torchvision.transforms import ToTensor, Compose, Resize

from models import LeNet5
from common import DistributedTrainer, DistributedTester

epochs = 30
batch_size, num_processes = 32, 5
def get_datasets():
    transforms = Compose([Resize((32, 32)), ToTensor()])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transforms)
    test_dataset = MNIST(root="./data", train=True, download=True, transform=transforms)
    return train_dataset, test_dataset

def get_model():
    return LeNet5()

def get_optimizer(model):
    return Adam(model.parameters(), 1e-3)

def get_scheduler(optimizer):
    return MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

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
