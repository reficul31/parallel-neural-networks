import torch

from .server import Worker
from torch.optim.optimizer import Optimizer
from .utils import MessageCode, send_message, ravel_model_params

class Hogwild(Optimizer):
    """
    Custom Hogwild Optimizer to perform distributed
    parallel Stochastic Gradient Descent
    """
    def __init__(self, params, lr, n_push, n_pull, model):
        """
        Initialize an instance of the Hogwild optimizer

        @param params - Parameters for the model
        @param lr - Learning Rate for the optimizer
        @param n_push - Number of steps after which we push gradients
        @param n_pull - Number of steps after which we request parameters
        @param model - Instance of the model
        """
        defaults = dict(lr=lr,)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.accumulated_gradients = torch.zeros(ravel_model_params(model).size()).to(device)
        self.n_pull = n_pull
        self.n_push = n_push

        self.model = model
        send_message(MessageCode.ParameterUpdate, ravel_model_params(self.model))
        self.idx = 0

        listener = Worker(self.model)
        listener.start()

        super(Hogwild, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Overriding the step function for the optimizer. In this function
        we perform distributed SGD. After n_pull and n_push steps we 
        request and send gradients to and from the server respectively.

        @param closure - Closure to calculate the loss
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        if self.idx % self.n_pull == 0:
            send_message(MessageCode.ParameterRequest, self.accumulated_gradients) # dummy val 

        lr = self.param_groups[0]['lr']
        gradients = ravel_model_params(self.model, grads=True)
        self.accumulated_gradients.add_(-lr, gradients)

        if self.idx % self.n_push == 0:
            send_message(MessageCode.GradientUpdate, self.accumulated_gradients)
            self.accumulated_gradients.zero_()

        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                d_parameter = parameter.grad.data
                parameter.data.add_(-group['lr'], d_parameter)
        
        self.idx += 1
        return loss
