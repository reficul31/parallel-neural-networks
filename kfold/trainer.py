import os
import time
import torch
import numpy as np

class Trainer(object):
    def __init__(self, criterion, root_dir, batch_size):
        self.criterion = criterion
        self.root_dir = root_dir
        self.batch_size = batch_size
    
    def __call__(self, fold, model, optimizer, scheduler, data_loader, epochs=100, save_checkpoint_frequency=20, print_frequency=15):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loss = []

        model_dir = os.path.join(self.root_dir, str(fold))
        for epoch in range(epochs):
            model.train()
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
            
            train_loss.append(np.mean(log_loss))
            if epoch % save_checkpoint_frequency == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, os.path.join(model_dir, "checkpoint_{}.tar".format(epoch)))
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, os.path.join(model_dir, "checkpoint_latest.tar"))
                np.save(os.path.join(model_dir, "train-loss-epoch-{}.npy".format(epoch)), train_loss)