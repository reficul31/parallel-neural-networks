import os
import time
import torch
import numpy as np
from torch.multiprocessing import Process

from torch.utils.data import DataLoader, DistributedSampler
from .trainer import Trainer

class DistributedTester(object):
    def __init__(self, data_loader, root_dir, num_classes = 10):
        self.data_loader = data_loader
        self.root_dir = root_dir
        self.num_classes = num_classes
    
    def __call__(self, model):
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
    def __init__(self, num_processes, criterion, root_dir, batch_size):
        self.criterion = criterion
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_processes = num_processes
    
    def __call__(self, model, dataset, optimizer, scheduler, epochs=100, save_checkpoint_frequency=20, print_frequency=15):
        model.share_memory()
        processes = []
        for rank in range(self.num_processes):
            data_loader = DataLoader(dataset, sampler=DistributedSampler(dataset=dataset, num_replicas=self.num_processes, rank=rank))

            processes.append(Process(target=self.train, args=(rank, model, optimizer, scheduler, data_loader, epochs, save_checkpoint_frequency, print_frequency)))
        
        for p in processes:
            p.start()
        for p in processes:
            p.join()
    
    def train(self, model, optimizer, scheduler, data_loader, epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loss = []

        model.train()
        for _ in range(epochs):
            _ = len(data_loader.dataset) / self.batch_size
            
            log_loss = []
            for _, (X, y) in enumerate(data_loader):
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = self.criterion(outputs, y)

                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                log_loss.append(loss.item())
            train_loss.append(np.mean(log_loss))
