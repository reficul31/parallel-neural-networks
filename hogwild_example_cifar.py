import os
import sys
import torch
import pandas as pd
import torch.nn.functional as F
import torch.distributed as dist

from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score

from models import AlexNet
from hogwild import Hogwild, Server

epochs = 10
lr, batch_size = 1e-3, 32
num_push, num_pull = 5, 5

transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def evaluate(net, testloader):
    net.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            test_loss += F.cross_entropy(outputs, labels).item()

    test_accuracy = accuracy_score(predicted, labels)
    return test_loss, test_accuracy

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception("Need arguments for server and worker")
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=int(sys.argv[1]), world_size=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    net = AlexNet().to(device)
    if int(sys.argv[1]) == 0:
        server = Server(net)
        server.run()
    
    optimizer = Hogwild(net.parameters(), lr=lr, n_push=num_push, n_pull=num_pull, model=net)
    scheduler = ReduceLROnPlateau(optimizer, patience=1, verbose=True, min_lr=1e-3)

    logs = []
    net.train()
    for epoch in range(epochs):
        print("Training for epoch {}".format(epoch))
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(predicted, labels)

            log_obj = {
                'timestamp': datetime.now(),
                'iteration': i,
                'training_loss': loss.item(),
                'training_accuracy': accuracy,
            }

            
            log_obj['test_loss'], log_obj['test_accuracy']= evaluate(net, testloader)
            print("Timestamp: {timestamp} | "
                    "Iteration: {iteration:6} | "
                    "Loss: {training_loss:6.4f} | "
                    "Accuracy : {training_accuracy:6.4f} | "
                    "Test Loss: {test_loss:6.4f} | "
                    "Test Accuracy: {test_accuracy:6.4f}".format(**log_obj))

            logs.append(log_obj)
                
        val_loss, val_accuracy = evaluate(net, testloader, verbose=True)
        scheduler.step(val_loss)

    df = pd.DataFrame(logs)
    print(df)
    df.to_csv('log/node{}.csv'.format(dist.get_rank()), index_label='index')
