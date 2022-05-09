import os
import time
import torch
import numpy as np
import pandas as pd

from torch.multiprocessing import Process
from torch.utils.data import DataLoader, DistributedSampler

class DistributedTester(object):
    """
    Distributed Tester object is used to perform distributed
    testing on the dataset provided in the arguments
    """
    def __init__(self, data_loader, root_dir, num_classes = 10):
        """
        Initialize the Distributed Tester object

        @param data_loader - Data loader for the test dataset
        @param root_dir - Directory to save the results
        @param num_classes - Number of classes in the dataset
        """
        self.data_loader = data_loader
        self.root_dir = root_dir
        self.num_classes = num_classes
    
    def __call__(self, model):
        """
        Call function for the Distributed tester object. Saves the 
        results of the test in the form of a confusion matrix in the
        root_dir passed in the initilization of the caller.

        @param model - The model to test with the dataset
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        with torch.set_grad_enabled(False):
            for _, (X, y) in enumerate(self.data_loader):
                X = X.to(device)
                y = y.to(device)
                outputs = model(X)
                _, predicted = torch.max(outputs, 1)
                for t, p in zip(y.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        np.save(os.path.join(self.root_dir, "confusion_matrix.npy"), confusion_matrix)

class DistributedTrainer(object):
    """
    Distributed Trainer object is used to perform distributed
    training on the dataset provided in the arguments. Arguments
    provided in the initialization and the caller function are
    used as hyperparameters for training.
    """
    def __init__(self, num_processes, criterion, root_dir, batch_size):
        """
        Initialize the Distributed Trainer object

        @param num_processes - Number of processes which need to be spawned
        @param criterion - Criterion used for training the model
        @param root_dir - Directory to save the results
        @param batch_size - Batch size to initialize the data loader
        """
        self.criterion = criterion
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_processes = num_processes
    
    def __call__(self, model, dataset, optimizer, scheduler, epochs=100, print_frequency=20):
        """
        Call function for the Distributed Trainer object. Spawns
        mmultiple processes which perform distributed training
        on the model placed in shared memory.

        @param model - The model to train with the dataset
        @param dataset - Train dataset for the model
        @param optimizer - Optimizer for training the model
        @param scheduler - Scheduler for training the model
        @param epochs - Number of epochs to train the model
        @param print_frequency - Print training progress intervals
        """
        model.share_memory()
        processes = []
        for rank in range(self.num_processes):
            data_loader = DataLoader(dataset, sampler=DistributedSampler(dataset=dataset, num_replicas=self.num_processes, rank=rank))
            processes.append(Process(target=self.train, args=(rank, model, optimizer, scheduler, data_loader, epochs, print_frequency)))
        
        for p in processes:
            p.start()
        for p in processes:
            p.join()
    
    def train(self, rank, model, optimizer, scheduler, data_loader, epochs, print_frequency):
        """
        Train job to be performed for distributed training on the dataset.
        Each process runs this function with its particular data shard.

        @param rank - Rank of the current process
        @param model - Common model instance in shared memory
        @param optimizer - Optimizer used for training
        @param scheduler - Scheduler for training the model
        @param data_loader - Data loader for the process data shard
        @epochs - Number of epochs to train the model
        @print_frequency - Print training progress intervals
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loss, times = [], []

        model.train()
        for epoch in range(epochs):
            batch_step_size = len(data_loader.dataset) / self.batch_size
            
            log_loss = []
            start = time.time()
            for batch_idx, (X, y) in enumerate(data_loader):
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = self.criterion(outputs, y)

                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                log_loss.append(loss.item())
                if batch_idx % print_frequency == 0:
                    print("Epoch {} : Worker - {} ({:04d}/{:04d}) Loss = {:.4f}".format(epoch + 1, rank, batch_idx, int(batch_step_size), loss.item()))
            
            times.append(time.time())
            train_loss.append(np.mean(log_loss))
            print("Epoch {} done: Time = {}, Mean Loss = {}".format(epoch + 1, time.time() - start, train_loss[-1]))
        
        df = pd.DataFrame({'train_loss': train_loss, 'time': times})
        df.to_csv(os.path.join(self.root_dir, "train_data_{}.csv".format(rank)))