import torch
import torch.optim
import torch.distributed as dist

from threading import Thread
from .utils import MessageCode, send_message, ravel_model_params, unravel_model_params

class Worker(Thread):
    """
    Worker thread that runs the training node and requests
    the server for model parameters and sends gradients
    """
    def __init__(self, model):
        """
        Initialize an instance of the worker thread

        @param model - Instance of the model
        """
        self.model = model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_parameters = torch.zeros(ravel_model_params(model).numel() + 2).to(device)
        super(Worker, self).__init__()
    
    def run(self):
        """
        Function to start the worker thread. The worker thread
        waits for the message from the server and then updates
        the parameters of the local instance of the model
        """
        self.running = True
        while self.running:
            dist.recv(tensor=self.model_parameters)
            if MessageCode(self.model_parameters[1].item()) == MessageCode.ParameterUpdate:
                unravel_model_params(self.model, self.model_parameters[2:])

class Server(Thread):
    """
    Server thread that runs the parameter server and gets
    requests for parameter upates and gradient updates from
    the worker nodes training the model
    """
    def __init__(self, model):
        """
        Initialize an instance of the server thread

        @param model - Instance of the model
        """
        self.model = model
        self.parameter_shard = torch.rand(ravel_model_params(model).numel())
        self.model_parameters = torch.zeros(ravel_model_params(model).numel() + 2)
        super(Server, self).__init__()

    def run(self):
        """
        Function to start the server thread. The server thread
        waits for the message from the worker nodes. Based on the
        message recieved it either updates the parameters or sends
        the paramters to the worker or updates the gradients.
        """
        self.running = True
        while self.running:
            dist.recv(tensor=self.model_parameters)
            if MessageCode(self.model_parameters[1].item()) == MessageCode.ParameterUpdate:
                self.parameter_shard = self.model_parameters[2:].clone()

            elif MessageCode(self.model_parameters[1].item()) == MessageCode.ParameterRequest:
                send_message(MessageCode.ParameterUpdate, self.parameter_shard, dst=int(self.model_parameters[0].item()))    

            elif MessageCode(self.model_parameters[1].item()) == MessageCode.GradientUpdate:
                self.parameter_shard.add_(self.model_parameters[2:])
