import os

from .jobscheduler import JobScheduler
from .waiter import Waiter

class KFold(object):
    def __init__(self, kfolds, train_dataset, root_dir, criterion, batch_size, parallelize=True):
        self.waiter = Waiter(train_dataset, criterion, batch_size, kfolds)
        self.root_dir = root_dir
        self.parallelize = parallelize

    def __call__(self, models, schedulers, get_optimizer, trainer_params):
        for net in models:
            name = net.__name__
            if not os.path.exists(os.path.join(self.root_dir, name)):
                os.mkdir(os.path.join(self.root_dir, name))
            
            for scheduler_name in schedulers:
                root_dir = os.path.join(self.root_dir, name, scheduler_name)
                if not os.path.exists(root_dir):
                    os.mkdir(root_dir)
                
                waiter_params = {
                    "net": net,
                    "get_optimizer": get_optimizer,
                    "scheduler_name": scheduler_name,
                    "root_dir": root_dir,
                    "trainer_params": trainer_params
                }
                
                job_scheduler = JobScheduler()
                job_scheduler(self.waiter, waiter_params, self.parallelize)