from torch.multiprocessing import Process

def job(params):
    trainer, trainer_params, tester, tester_params = params
    trainer(**trainer_params)
    tester(**tester_params)

class JobScheduler(object):
    def __init__(self, job = job):
        self.job = job
    
    def __call__(self, waiter, waiter_params, parallelize=True):
        if parallelize:
            processes = []
            for params in waiter(**waiter_params):
                p = Process(target=self.job, args=(params,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            for params in waiter(**waiter_params):
                self.job(params)