import torch
import torch.distributed as dist

from enum import Enum

class MessageCode(Enum):
    ParameterRequest = 0
    GradientUpdate = 1
    ParameterUpdate = 2
    EvaluateParams = 3

def ravel_model_params(model, grads=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m_parameter = torch.Tensor([0]).to(device)
    for parameter in list(model.parameters()):
        if grads:
            m_parameter = torch.cat((m_parameter, parameter.grad.view(-1)))
        else:
            m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
    return m_parameter[1:]

def unravel_model_params(model, parameter_update):
    current_index = 0 # keep track of where to read from parameter_update
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        parameter.data.copy_(parameter_update[current_index:current_index+numel].view(size))
        current_index += numel

def send_message(message_code, payload, dst=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m_parameter = torch.Tensor([dist.get_rank(), message_code.value]).to(device)
    m_parameter = torch.cat((m_parameter, payload))
    dist.isend(tensor=m_parameter, dst=dst)
