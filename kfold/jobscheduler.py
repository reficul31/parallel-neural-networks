from torch.multiprocessing import Process

def job(params):
    trainer, trainer_params, tester, tester_params = params
    trainer(**trainer_params)
    tester(**tester_params)

class JobScheduler(object):
    """
    JobScheduler object is used to scheduling th execution of various jobs
    """
    def __init__(self, job = job):
        """
        Initialize the JobScheduler object

        @param job - Job to be scheduled.
        """
        self.job = job
    
    def __call__(self, waiter, waiter_params, parallelize=True):
        """
        Call function for Job Scheduler. This method is responsible for 
        creating the required number of processes mentioned in the waiter_params.
        If parallelize parameter is set to False then it will run the required job sequentially.
        
        @param waiter - Object of waiter class
        @param waiter_params - A dictionary of key-value pairs specifying the parameters like network,optimizer to be used, scheduler name, root directory location and trainer's parameters.
        @param parallelize - Boolean specifying whether to run sequentially or parallelly. If True then the job is run parallely else it is run sequentially.
        """
        if parallelize:
            processes = [Process(target=self.job, args=(params,)) for params in waiter(**waiter_params)]
            for p in processes:
                p.start()
                
            for p in processes:
                p.join()
        else:
            for params in waiter(**waiter_params):
                self.job(params)