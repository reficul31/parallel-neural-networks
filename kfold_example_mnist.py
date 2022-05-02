import os
import torch

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torch.multiprocessing import set_start_method
from torchvision.transforms import ToTensor, Resize, Compose

from models import LeNet5
from kfold import Waiter, JobScheduler


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.abspath(__file__))
    set_start_method('spawn', force=True)

    def get_models():
        yield LeNet5

    def get_optimizer(model):
        return Adam(model.parameters(), 1e-3)

    transforms = Compose([Resize((32, 32)), ToTensor()])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transforms)

    trainer_params = dict({"epochs": 2})
    waiter = Waiter(['cosine'], CrossEntropyLoss(), root_dir, splits=2, trainer_params = trainer_params)
    waiter_params = {"train_dataset": train_dataset, "get_models": get_models, "get_optimizer": get_optimizer}

    job_scheduler = JobScheduler()
    job_scheduler(waiter, waiter_params)