import uuid
import contextlib

from enum import Enum, auto
from threading import Thread, Event

from training import execute_episode, evaluate_neural_network, duel_between_neural_networks


class WorkType:
    EXECUTE_EPISODE = auto()
    DUEL_BETWEEN_NEURAL_NETWORKS = auto()
    EVALUATE_NEURAL_NETWORK = auto()


class Worker:
    def __init__(self):
        self._executor_thread = None
        self._results = None

    def setup(self):
        pass

    def run(self, work_type, iterations, *args, **kwargs):
        self._results = []
        self._executor_thread = Thread(name=self.get_executor_thread_name(), target=self._run, 
                                       args=(work_type, iterations, args, kwargs))
        self._executor_thread.start()

    @staticmethod    
    def execute_episode(*args, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def duel_between_neural_networks(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def evaluate_neural_network(*args, **kwargs):
        raise NotImplementedError

    def teardown(self):
        pass
    
    def wait(self):
        return self._executor_thread.join() if self._executor_thread else None
    
    def get_results(self):
        return self._results

    def _run(self, work_type, iterations, args, kwargs):
        target = self.get_target(work_type)
        self.setup()
        for _ in range(iterations):
            result = target(*args, **kwargs)
            self._results.append(result)
    
    def get_executor_thread_name(self):
        id_ = uuid.uuid4()
        id_ = str(id_).split('-', 1)[0]
        return f'{self.__class__.__name__}-{id_}'
    
    @classmethod
    def get_target(cls, work_type):
        if work_type is WorkType.EXECUTE_EPISODE:
            return cls.execute_episode
        elif work_type is WorkType.DUEL_BETWEEN_NEURAL_NETWORKS:
            return cls.duel_between_neural_networks
        elif work_type is WorkType.EVALUATE_NEURAL_NETWORK:
            return cls.evaluate_neural_network
        raise TypeError('expecting WorkType object')


class ThreadWorker(Worker):
    @staticmethod
    def execute_episode(*args, **kwargs):
        return execute_episode(*args, **kwargs)
    
    @staticmethod
    def evaluate_neural_network(*args, **kwargs):
        return evaluate_neural_network(*args, **kwargs)
    
    @staticmethod
    def duel_between_neural_networks(*args, **kwargs):
        return duel_between_neural_networks(*args, **kwargs)


class GoogleCloudWorker(Worker):
    def __init__(self):
        pass


class WorkerManager:
    def __init__(self):
        self._workers = []
        self._waiter_thread = None
        self._finished_event = Event()
    
    @contextlib.contextmanager
    def run(self, work_type, iterations, *args, **kwargs):
        if isinstance(work_type, WorkType):
            raise TypeError('expecting WorkerType object')
        self._finished_event.clear()
        worker_iterations = WorkerManager.divide_iterations(iterations, len(self._workers))
        for worker, total_iterations in zip(self._workers, worker_iterations):
            worker.run(work_type, total_iterations, *args, **kwargs)
        self._waiter_thread = Thread(target=self._wait_workers)
        self._waiter_thread.start()
        
        try:
            yield self._finished_event
        finally:
            for worker in self._workers:
                worker.teardown()
    
    def _wait_workers(self):
        for worker in self._workers:
            worker.wait()
        self._finished_event.set()
    
    def get_results(self):
        results = []
        for worker in self._workers:
            results.extend(worker.get_results())
        return results

    def add_worker(self, worker):
        if not isinstance(worker, Worker):
            raise TypeError('expecting Worker object')
        self._workers.append(worker)
    
    def total_workers(self):
        return len(self._workers)

    @staticmethod
    def divide_iterations(total_iterations, total_workers):
        worker_total_iterations = [0] * total_iterations
        for i in range(total_iterations):
            worker_total_iterations[i % total_workers] += 1
        return  worker_total_iterations
