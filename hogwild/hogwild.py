import torch
from torch.optim.optimizer import Optimizer

from .server import Worker
from .utils import MessageCode, send_message, ravel_model_params

class Hogwild(Optimizer):
    def __init__(self, params, lr, n_push, n_pull, model):
        defaults = dict(lr=lr,)
        self.accumulated_gradients = torch.zeros(ravel_model_params(model).size())
        self.n_pull = n_pull
        self.n_push = n_push

        self.model = model
        # this sets the initial model parameters
        send_message(MessageCode.ParameterUpdate, ravel_model_params(self.model))
        self.idx = 0

        listener = Worker(self.model)
        listener.start()

        super(Hogwild, self).__init__(params, defaults)

    def step(self, closure=None):
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
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
        
        self.idx += 1
        return loss
