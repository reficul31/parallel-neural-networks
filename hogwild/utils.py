import torch
import torch.distributed as dist

from enum import Enum

class MessageCode(Enum):
    """
    Enum for passing messages between different processes
    """
    ParameterRequest = 0
    GradientUpdate = 1
    ParameterUpdate = 2
    EvaluateParams = 3

def ravel_model_params(model, gradients=False):
    """
    Flatten the model parameters into a single tensor. This
    is done so that we can send the tensor through the 
    multiprocessing pipeline

    @param model - Model to get the shape of the paramters
    @param gradients - Which argument to flatten gradients or parameters

    @returns - Flattened model parameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_parameters = torch.Tensor([0]).to(device)
    for parameter in list(model.parameters()):
        if gradients:
            model_parameters = torch.cat((model_parameters, parameter.grad.view(-1)))
        else:
            model_parameters = torch.cat((model_parameters, parameter.data.view(-1)))
    return model_parameters[1:]

def unravel_model_params(model, parameter_update):
    """
    Unflatten the model paramters into the model param tensor shape.
    This is done after the thread recieves the message from another
    process.

    @param model - Model to get the shape of the paramters
    @param parameter_update - Clone of the model parameters to be copied
    """
    current_index = 0
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        parameter.data.copy_(parameter_update[current_index:current_index+numel].view(size))
        current_index += numel

def send_message(message_code, payload, destination=0):
    """
    Send messages between different processes with the specified payload.

    @param messagee_code - Message code of the message
    @param payload - Payload of the message to be sent
    @param destination - Destination of the message
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_parameters = torch.Tensor([dist.get_rank(), message_code.value]).to(device)
    model_parameters = torch.cat((model_parameters, payload))
    dist.isend(tensor=model_parameters, dst=destination)
