import os
import time
import json

from .jobscheduler import JobScheduler
from .waiter import Waiter

class AsyncKFold(object):
    """
    AsyncKFold object is used to perform K Fold cross validation on a list of models and schedulers
    on a given dataset.
    """
    def __init__(self, kfolds, train_dataset, root_dir, criterion, batch_size, parallelize=True):
        """
        Initialize AsyncKFold object.

        @param kfolds - Number of folds to be used.
        @param train_dataset - object containing the training data
        @param root_dir - String specifying the absolute path of the root directory
        @param criterion - object of the criterion to be used while training
        @param batch_size - Batch size for training
        @param parallelize - Boolean specifying whether job should be run sequentially or parallely. If True job is executed parallely else sequentially.
        """
        
        self.waiter = Waiter(train_dataset, criterion, batch_size, kfolds)
        self.root_dir = root_dir
        self.parallelize = parallelize

    def __call__(self, models, schedulers, get_optimizer, trainer_params):
        """
        Call function for AsyncKFold object. Creates directories for differenr models and their schedulers.
        Instantiates Job scheduler and calculates the total execution time.

        @param models: List of model objects to be used
        @param schedulers - List of strings specifying the names of the schedulers
        @param get_optimizer - function that takes model object as input and returns the optimizer to be used.
        @param trainer_params - Dictionary of key value pairs specifying the epochs,save checkpoint frequency and print frequency.
        """

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
                start = time.time()
                job_scheduler(self.waiter, waiter_params, self.parallelize)
                end = time.time()

                with open(os.path.join(root_dir, 'time.json'), 'w') as time_file:
                    time_file.write(json.dumps(dict({'start': start, 'end': end})))
