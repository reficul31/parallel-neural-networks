import torch
import torch.optim
import torch.distributed as dist

from threading import Thread
from .utils import MessageCode, send_message, ravel_model_params, unravel_model_params

class Worker(Thread):
    def __init__(self, model):
        self.model = model
        self.m_parameter = torch.zeros(ravel_model_params(model).numel() + 2)
        super(Worker, self).__init__()

    def receive(self, message_code, parameter):
        if message_code == MessageCode.ParameterUpdate:
            unravel_model_params(self.model, parameter)
    
    def run(self):
        self.running = True
        while self.running:
            dist.recv(tensor=self.m_parameter)
            self.receive(MessageCode(self.m_parameter[1].item()), self.m_parameter[2:])

class Server(Thread):
    def __init__(self, model):
        self.model = model
        self.parameter_shard = torch.rand(ravel_model_params(model).numel())
        self.m_parameter = torch.zeros(ravel_model_params(model).numel() + 2)
        super(Server, self).__init__()

    def receive(self, sender, message_code, parameter):
        print("Processing message: {} from sender {}".format(message_code.name, sender))
        if message_code == MessageCode.ParameterUpdate:
            self.parameter_shard = parameter.clone()

        elif message_code == MessageCode.ParameterRequest:
            send_message(MessageCode.ParameterUpdate, self.parameter_shard, dst=sender)    

        elif message_code == MessageCode.GradientUpdate:
            self.parameter_shard.add_(parameter)
    
    def run(self):
        self.running = True
        while self.running:
            dist.recv(tensor=self.m_parameter)
            self.receive(int(self.m_parameter[0].item()),
                         MessageCode(self.m_parameter[1].item()),
                         self.m_parameter[2:])
