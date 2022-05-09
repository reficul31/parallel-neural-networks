import os
import sys
import time
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

from models import VGG
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

    test_accuracy = accuracy_score(predicted.cpu().detach().numpy(), labels.cpu().detach().numpy())
    return test_loss, test_accuracy

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception("Need 3 arguments for server and worker")
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=int(sys.argv[1]), world_size=int(sys.argv[2]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    net = VGG().to(device)
    if int(sys.argv[1]) == 0:
        server = Server(net)
        server.run()
    
    optimizer = Hogwild(net.parameters(), lr=lr, n_push=num_push, n_pull=num_pull, model=net)
    scheduler = ReduceLROnPlateau(optimizer, patience=1, verbose=True, min_lr=1e-3)

    logs = []
    net.train()
    for epoch in range(epochs):
        print("Training for epoch {}".format(epoch))
        batch_step_size = len(trainloader.dataset) / batch_size

        start = time.time()
        for batch_idx, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 20:
                print("Epoch {} : Worker - {} ({:04d}/{:04d}) Loss = {:.4f}".format(epoch + 1, dist.get_rank(), batch_idx, int(batch_step_size), loss.item()))
                _, predicted = torch.max(outputs, 1)
                accuracy = accuracy_score(predicted.cpu().detach().numpy(), labels.cpu().detach().numpy())

                log_obj = {
                    'timestamp': datetime.now(),
                    'iteration': batch_idx,
                    'training_loss': loss.item(),
                    'training_accuracy': accuracy,
                }
                
                log_obj['test_loss'], log_obj['test_accuracy']= evaluate(net, testloader)
                logs.append(log_obj)
                
        val_loss, val_accuracy = evaluate(net, testloader, verbose=True)
        scheduler.step(val_loss)
        print("Epoch {} done: Time = {}, Val Loss = {}, Val Accuracy = {}".format(epoch + 1, time.time() - start, val_loss, val_accuracy))

    df = pd.DataFrame(logs)
    print(df)
    df.to_csv('log/node{}.csv'.format(dist.get_rank()), index_label='index')
