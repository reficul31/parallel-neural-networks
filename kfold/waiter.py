import os
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler

from kfold import get_scheduler, Trainer, Tester

class Waiter(object):
    def __init__(self, train_dataset, criterion, batch_size, splits):
        self.splits = splits
        self.train_dataset = train_dataset
        self.criterion = criterion
        self.batch_size = batch_size
    
    def __call__(self, net, get_optimizer, scheduler_name, root_dir, trainer_params=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        splitter = KFold(n_splits=self.splits, shuffle=True)
        for fold, (train_ids, test_ids) in enumerate(splitter.split(self.train_dataset)):
            if not os.path.exists(os.path.join(root_dir, str(fold))):
                os.mkdir(os.path.join(root_dir, str(fold)))
            
            train_subsampler, test_subsampler = SubsetRandomSampler(train_ids), SubsetRandomSampler(test_ids)
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_subsampler)
            test_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=test_subsampler)

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
                **trainer_params
            }

            tester = tester = Tester(test_loader, root_dir)
            tester_args = {
                "fold": fold,
                "model": model
            }
            yield trainer, trainer_args, tester, tester_args