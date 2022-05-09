import os
import time
import torch
import numpy as np

class Trainer(object):
    """
    Trainer object is used to perform training on the dataset 
    provided in the arguments. Arguments provided in the 
    initialization and the caller function are used as 
    hyperparameters for training.
    """
    def __init__(self, criterion, root_dir, batch_size, save_latest=True):
        """
        Initialize the Trainer object

        @param criterion - Criterion used for training the model
        @param root_dir - Directory to save the results
        @param batch_size - Batch size to initialize the data loader
        @param save_latest - Boolean flag whether to save the latest model
        """
        self.criterion = criterion
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.save_latest = save_latest
    
    def __call__(self, fold, model, optimizer, scheduler, data_loader, epochs=100, save_checkpoint_frequency=20, print_frequency=15):
        """
        Call function for the Trainer object. The function defines
        the train job to be performed for each and every fold of the
        model. This function also saves the training loss and model
        checkpoint at every epoch. This enables the model to do restarts
        by loading the latest checkpoint if the training is stopped.

        @param fold - Current fold that is being executed
        @param model - The model to train with the dataset
        @param optimizer - Optimizer for training the model
        @param scheduler - Scheduler for training the model
        @param data_loader - Data loader for the training dataset
        @param epochs - Number of epochs to train the model
        @param save_checkpoint_frequency - Interval to save checkpoint of model
        @param print_frequency - Print training progress intervals
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loss = []

        model_dir = os.path.join(self.root_dir, str(fold))
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
                    print("Epoch {} : Worker - {} ({:04d}/{:04d}) Loss = {:.4f}".format(epoch + 1, fold, batch_idx, int(batch_step_size), loss.item()))
            
            train_loss.append(np.mean(log_loss))
            print("Epoch {} done: Time = {}, Mean Loss = {}".format(epoch + 1, time.time() - start, train_loss[-1]))
            if epoch % save_checkpoint_frequency == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, os.path.join(model_dir, "checkpoint_{}.tar".format(epoch)))
            elif self.save_latest:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, os.path.join(model_dir, "checkpoint_latest.tar"))
                np.save(os.path.join(model_dir, "train-loss-epoch-{}.npy".format(epoch)), train_loss)