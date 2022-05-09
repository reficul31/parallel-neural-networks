import os
import torch
import numpy as np

class Tester(object):
    """
    Tester object is used to perform testing 
    on the dataset provided in the arguments
    """
    def __init__(self, data_loader, root_dir, num_classes = 10):
        """
        Initialize the Tester object

        @param data_loader - Data loader for the test dataset
        @param root_dir - Directory to save the results
        @param num_classes - Number of classes in the dataset
        """
        self.data_loader = data_loader
        self.root_dir = root_dir
        self.num_classes = num_classes
    
    def __call__(self, fold, model):
        """
        Call function for the tester object. Saves the results 
        of the test in the form of a confusion matrix in the
        root_dir passed in the initilization of the caller.

        @param fold - Current fold that is being tested
        @param model - The model to test with the dataset
        """
        model_dir = os.path.join(self.root_dir, str(fold))
        if not os.path.exists(model_dir):
            raise Exception("Directory does not exist:", model_dir)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(os.path.join(model_dir, "checkpoint_latest.tar"))
        model.load_state_dict(checkpoint['model_state_dict'])

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
        np.save(os.path.join(model_dir, "confusion_matrix.npy"), confusion_matrix)