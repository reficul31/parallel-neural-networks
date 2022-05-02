import os
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler

from kfold import get_scheduler, Trainer, Tester

class Waiter(object):
    def __init__(self, schedulers, criterion, root_dir, splits=5, batch_size=64, trainer_params = None):
        self.root_dir = root_dir
        self.schedulers = schedulers
        self.criterion = criterion
        self.batch_size = batch_size
        self.splits = splits
        self.trainer_params = trainer_params
    
    def __call__(self, train_dataset, get_models, get_optimizer):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for net in get_models():
            name = net.__name__
            if not os.path.exists(os.path.join(self.root_dir, name)):
                os.mkdir(os.path.join(self.root_dir, name))
            for scheduler_name in self.schedulers:
                root_dir = os.path.join(self.root_dir, name, scheduler_name)
                if not os.path.exists(root_dir):
                        os.mkdir(root_dir)
    
                splitter = KFold(n_splits=self.splits, shuffle=True)
                for fold, (train_ids, test_ids) in enumerate(splitter.split(train_dataset)):
                    if not os.path.exists(os.path.join(root_dir, str(fold))):
                        os.mkdir(os.path.join(root_dir, str(fold)))
                    
                    train_subsampler, test_subsampler = SubsetRandomSampler(train_ids), SubsetRandomSampler(test_ids)
                    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_subsampler)
                    test_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=test_subsampler)

                    model = net()
                    model = model.to(device)

                    optimizer = get_optimizer(model)
                    scheduler = get_scheduler(scheduler_name, optimizer)

                    trainer = Trainer(self.criterion, root_dir, self.batch_size)
                    trainer_args = {
                        "fold": fold,
                        "model": model,
                        "optimizer": optimizer,
                        "scheduler": scheduler,
                        "data_loader": train_loader,
                        **self.trainer_params
                    }

                    tester = tester = Tester(test_loader, root_dir)
                    tester_args = {
                        "fold": fold,
                        "model": model
                    }
                    yield trainer, trainer_args, tester, tester_args