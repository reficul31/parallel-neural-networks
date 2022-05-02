from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torch.nn import CrossEntropyLoss
from torch.multiprocessing import set_start_method
from torchvision.transforms import ToTensor, Compose, RandomCrop, RandomHorizontalFlip, Normalize

from models import MobileNet, LeNet, VGG
from kfold import Waiter, JobScheduler

models = [MobileNet, LeNet, VGG]
def get_models():
    for model in models:
        yield model

def get_optimizer(model):
    return Adam(model.parameters(), 1e-3)

if __name__ == '__main__':
    root_dir = "."
    set_start_method('spawn', force=True)
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform_train)

    trainer_params = dict({"epochs": 10})
    waiter = Waiter(['warm'], CrossEntropyLoss(), root_dir, splits=5, trainer_params = trainer_params)
    waiter_params = {"train_dataset": train_dataset, "get_models": get_models, "get_optimizer": get_optimizer}

    job_scheduler = JobScheduler()
    job_scheduler(waiter, waiter_params)