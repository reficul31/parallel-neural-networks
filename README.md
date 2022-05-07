# Parallel Neural Networks

## Description of project
---

Stochastic Gradient Descent (SGD) is a commonly used optimization approach. A few
sophisticated strategies, including variance reduction, stochastic coordinate sampling, and
Nesterov's acceleration method, have been developed in recent years to speed up the
convergence of SGD. A natural question is whether these techniques can be implemented in
parallel on distributed systems to achieve speedup in training.
In this project, we explore some of the asynchronous stochastic gradient descent methods
such as HOGWILD. 

We aim to create a distributed task execution
environment that can be generalised for any type of data sharing task such as K-Fold
Cross-Validation or SGD. We will implement asynchronous stochastic gradient descent
method and benchmark their performance on different networks and datasets

## Description of Repository
---

In this repository we have included AsyncKFold and Asynschornous Stochastic Gradient Descent using HOGWILD menthod in pytorch. 


## Executing the code
---

```

```

## Testing on example code
---
- ### Testing Asynchornous KFold on MNIST dataset

```
python kfold_example_mnist.py 
```

 - ### Testing Asynchornous KFold on CIAFR dataset
```
 python kfold_example_cifar.py
 ```

## Observations 
---
 Most of the CPU is idle when a serial KFold is running. Our goal is to maximize the CPU utilization and decrease training time without having significant effect on the accuracy.

The CPU utilization for running LetNet, Mobile-Net and VGG serially for cosine and warm schedulers.

### CPU Utilization

![Serial CPU utilization](./asset/CPU-Usage-Non-Parallel.png)

As seen above most of the CPU while running KFold serially is not utilized. Due to this the time of execution is high.
The serial KFold took a total time of 165 minutes.


![Parallel CPU utilization](./asset/CPU-Usage-Parallel.png)

<b> Our parallel implementation was able to maximize CPU utilization resulting in almost 100% CPU utilization decreasing the execution time from 165 minutes to just 80 minutes </b>



### SpeedUp

![LeNet Speedup Graph](./asset/lenet-speedup.png)

As we see our parallel KFold implementation has a 2.72x speedup for cosine and warm schedulers.


<img src="./asset/class-accuracies.png" width="800px" height ="650px"  >
In the above graph we have shown the difference between the serial and parallel implementations of diferent models with different schedulers. 


### Accuracy

### The accuracy table obtained is as shown

| Module | Scheduler | Parallel Accuracy | Serial Accuracy |
--- | --- | --- | --- 
 | Mobile-Net | warm  | 0.7393 | 0.7344
 | Mobile-Net | cosine | 0.7394 | 0.7334
 | LeNet | warm| 0.5634 | 0.5399
 | LeNet | cosine | 0.5359 | 0.5682
 | VGG | warm | 0.7958 | 0.7810
 | VGG | cosine | 0.7719 | 0.7538

 As seen from the abive table and graph our parallel Asynchronous KFold is able to achieve similar accuracy as the serial KFold.


 

