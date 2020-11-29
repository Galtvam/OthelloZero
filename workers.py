import os
import uuid
import logging
import tempfile
import contextlib
import humanfriendly

from enum import Enum
from threading import Thread, Event

from pickle_training import pack_arguments_to_pickle, unpack_base64_pickle
from gcloud import get_instance, ssh_connection, get_instance_external_ip, \
        get_instance_internal_ip, SSH_USER
from training import execute_episode, evaluate_neural_network, \
    duel_between_neural_networks


class WorkType:
    EXECUTE_EPISODE = 'Execute Episode'
    DUEL_BETWEEN_NEURAL_NETWORKS = 'Duel between Neural Networks'
    EVALUATE_NEURAL_NETWORK = 'Evaluate Neural Network'


class Worker:
    def __init__(self):
        self._executor_thread = None
        self._results = None
        self._worker_manager = None

    def setup(self, work_type, iterations, *args, **kwargs):
        pass

    def run(self, work_type, iterations, *args, **kwargs):
        self._results = []
        self._executor_thread = Thread(name=self.get_executor_thread_name(), target=self._run, 
                                       args=(work_type, iterations, args, kwargs))
        self._executor_thread.start()

    def execute_episode(self, *args, **kwargs):
        raise NotImplementedError
    
    def duel_between_neural_networks(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate_neural_network(self, *args, **kwargs):
        raise NotImplementedError

    def teardown(self, work_type):
        pass
    
    def wait(self):
        return self._executor_thread.join() if self._executor_thread else None
    
    def get_results(self):
        return self._results

    def _run(self, work_type, iterations, args, kwargs):
        target = self.get_target(work_type)
        self.setup(work_type, iterations, *args, *kwargs)
        for i in range(1, iterations + 1):
            logging.info(f'Task {work_type} ({i}/{iterations}): Starting...')
            result = target(*args, **kwargs)
            self._results.append(result)
            logging.info(f'Task {work_type} ({i}/{iterations}): Finished!')
        self.teardown(work_type)
    
    def get_executor_thread_name(self):
        id_ = uuid.uuid4()
        id_ = str(id_).split('-', 1)[0]
        return f'{self.__class__.__name__}-{id_}'
    
    def get_target(self, work_type):
        if work_type is WorkType.EXECUTE_EPISODE:
            return self.execute_episode
        elif work_type is WorkType.DUEL_BETWEEN_NEURAL_NETWORKS:
            return self.duel_between_neural_networks
        elif work_type is WorkType.EVALUATE_NEURAL_NETWORK:
            return self.evaluate_neural_network
        raise TypeError('expecting WorkType object')


class ThreadWorker(Worker):
    def execute_episode(self, *args, **kwargs):
        return execute_episode(*args, **kwargs)
    
    def evaluate_neural_network(self, *args, **kwargs):
        return evaluate_neural_network(*args, **kwargs)
    
    def duel_between_neural_networks(self, *args, **kwargs):
        return duel_between_neural_networks(*args, **kwargs)


class GoogleCloudWorker(Worker):
    SSH_PRIV_KEY = f'/home/{SSH_USER}/.ssh/{SSH_USER}-internal'
    SSH_PUB_KEY = f'{SSH_PRIV_KEY}.pub'

    def __init__(self, compute, project, zone, instance_name, key_filename):
        instance = get_instance(compute, project, zone, instance_name)
        if not instance:
            raise RuntimeError(f'Instance {instance_name} not found')
        self._instance = instance
        self._key_filename = key_filename
        
        self._internal_ssh_pub_key = None

        self._neural_network_weights_file = []
        self._ssh = None
        self._sftp = None
    
    def setup(self, work_type, iterations, *args, **kwargs):
        pass

    def execute_episode(self, board_size, neural_network, degree_exploration, 
                        num_simulations, policy_temperature, e_greedy):
        args = [board_size, self._neural_network_weights_file[0], degree_exploration, 
                num_simulations, policy_temperature, e_greedy]
        training_examples = self._remote_pickle_training_call('execute_episode', args)
        return training_examples
    
    def evaluate_neural_network(self, board_size, total_iterations, neural_network, num_simulations, degree_exploration, 
                                agent_class, agent_arguments):
        args = [board_size, total_iterations, self._neural_network_weights_file[0], 
                num_simulations, degree_exploration, agent_class, agent_arguments]
        net_wins = self._remote_pickle_training_call('evaluate_neural_network', args)
        return net_wins
    
    def duel_between_neural_networks(self, board_size, neural_network_1, neural_network_2, 
                                     degree_exploration, num_simulations):
        args = [board_size, self._neural_network_weights_file[0], 
                self._neural_network_weights_file[1], degree_exploration, num_simulations]
        net_wins = self._remote_pickle_training_call('duel_between_neural_networks', args)
        return net_wins

    def teardown(self, work_type):
        logging.info(f'Task {work_type} Teardown: Deleting cache files...')
        if self._neural_network_weights_file:
            self._sftp = self._ssh.open_sftp()

            for filepath in self._neural_network_weights_file:
                self._sftp.remove(filepath)
            self._ssh.close()

        self._ssh = None
        self._sftp = None
        self._neural_network_weights_file = []
    
    def _remote_pickle_training_call(self, command_name, args):
        args = pack_arguments_to_pickle(*args)
        command = 'docker run -v $PWD:/OthelloZero -v /tmp/:/tmp:ro igorxp5/othello-zero '
        command += f'OthelloZero/pickle_training.py {command_name} {" ".join(args)}'
        
        stdin, stdout, stderr = self._ssh.exec_command(command)
        stdout.channel.recv_exit_status()
        if stdout.channel.recv_exit_status() != 0:
            error = stderr.read().decode()
            logging.info(error)
            raise RuntimeError(error)
        
        return unpack_base64_pickle(stdout.readlines()[0].strip())


class WorkerManager:
    def __init__(self):
        self._workers = []
        self._waiter_thread = None
        self._finished_event = Event()
    
    def run(self, work_type, iterations, *args, **kwargs):
        if isinstance(work_type, WorkType):
            raise TypeError('expecting WorkerType object')
        self._finished_event.clear()
        worker_iterations = WorkerManager.divide_iterations(iterations, len(self._workers))
        self._setup(work_type, iterations, *args, **kwargs)
        for worker, total_iterations in zip(self._workers, worker_iterations):
            worker.run(work_type, total_iterations, *args, **kwargs)
        self._waiter_thread = Thread(target=self._wait_workers)
        self._waiter_thread.start()
        self._finished_event.wait()

    def get_results(self):
        results = []
        for worker in self._workers:
            results.extend(worker.get_results())
        return results

    def add_worker(self, worker):
        if not isinstance(worker, Worker):
            raise TypeError('expecting Worker object')
        worker._worker_manager = self 
        self._workers.append(worker)
    
    def total_workers(self):
        return len(self._workers)
    
    def _wait_workers(self):
        for worker in self._workers:
            worker.wait()
        self._finished_event.set()
    
    def has_google_worker(self):
        return any(isinstance(worker, GoogleCloudWorker) for worker in self._workers)
    
    def _setup(self, work_type, iterations, *args, **kwargs):
        files_to_send = []
        if work_type is WorkType.EXECUTE_EPISODE and self.has_google_worker():
            _, filepath = tempfile.mkstemp(suffix='.h5')
            neural_network = args[1]
            neural_network.save_checkpoint(filepath)
            files_to_send.append(filepath)

        elif work_type is WorkType.EVALUATE_NEURAL_NETWORK and self.has_google_worker():
            _, filepath = tempfile.mkstemp(suffix='.h5')
            neural_network = args[2]
            neural_network.save_checkpoint(filepath)
            files_to_send.append(filepath)
        
        elif work_type is WorkType.DUEL_BETWEEN_NEURAL_NETWORKS and self.has_google_worker():
            _, filepath = tempfile.mkstemp(suffix='.h5')
            neural_network_1 = args[1]
            neural_network_1.save_checkpoint(filepath)
            files_to_send.append(filepath)

            _, filepath = tempfile.mkstemp(suffix='.h5')
            neural_network_2 = args[2]
            neural_network_2.save_checkpoint(filepath)
            files_to_send.append(filepath)
        
        if self.has_google_worker():
            for filepath in files_to_send:
                file_size = humanfriendly.format_size(os.path.getsize(filepath))
                
                scp_processes = []
                uploaded_worker = None
                for worker in self._workers:
                    if isinstance(worker, GoogleCloudWorker):
                        worker._neural_network_weights_file.append(filepath)
                        ip = get_instance_external_ip(worker._instance)
                        worker._ssh = ssh_connection(ip, worker._key_filename)
                        if not uploaded_worker:
                            worker._sftp = worker._ssh.open_sftp()
                            logging.info(f'Uploading Neural network weights ({file_size})...')
                            worker._sftp.put(filepath, filepath)
                            logging.info(f'Neural network weights uploaded')
                            os.remove(filepath)
                            if not worker._internal_ssh_pub_key:
                                try:
                                    worker._sftp.stat(worker.SSH_PRIV_KEY)
                                except IOError:
                                    logging.info(f'Creating Internal SSH Key...')
                                    command = f'ssh-keygen -q -N "" -t rsa -f {worker.SSH_PRIV_KEY} -C {SSH_USER}'
                                    stdin, stdout, stderr = worker._ssh.exec_command(command)
                                    if stdout.channel.recv_exit_status() != 0:
                                        raise RuntimeError('cannot create internal ssh key')
                                    logging.info(f'Internal SSH Key created successfully!')
                                logging.info(f'Saving SSH Public Key...')
                                with worker._sftp.open(worker.SSH_PUB_KEY) as file:
                                    worker._internal_ssh_pub_key = file.read().decode('ascii')
                                logging.info(f'SSH Public Key saved!')
                            worker._sftp.close()
                            uploaded_worker = worker
                
                for worker in self._workers:
                    if isinstance(worker, GoogleCloudWorker) and uploaded_worker and uploaded_worker is not worker:
                        instance_name = worker._instance['name']
                        if not worker._internal_ssh_pub_key:
                            logging.info(f'Adding SSH Key to {instance_name}...')
                            worker._sftp = worker._ssh.open_sftp()
                            try:
                                worker._sftp.stat(worker.SSH_PUB_KEY)
                            except IOError:
                                command = f'echo "{uploaded_worker._internal_ssh_pub_key}" > {worker.SSH_PUB_KEY}'
                                stdin, stdout, stderr = worker._ssh.exec_command(command)
                                if stdout.channel.recv_exit_status() != 0:
                                    logging.error(stderr.read().decode())
                                    raise RuntimeError(f'cannot write pub key into {instance_name}')
                                command = f'cat {worker.SSH_PUB_KEY} >> /home/{SSH_USER}/.ssh/authorized_keys'
                                stdin, stdout, stderr = worker._ssh.exec_command(command)
                                if stdout.channel.recv_exit_status() != 0:
                                    logging.error(stderr.read().decode())
                                    raise RuntimeError(f'cannot add key to authorized_keys')
                            finally:
                                worker._sftp.close()
                            worker._internal_ssh_pub_key = uploaded_worker._internal_ssh_pub_key
                            logging.info(f'SSH Key added to {instance_name}!')
                        
                        ip = get_instance_internal_ip(worker._instance)
                        logging.info(f'Sending Neural network weights to instance: {instance_name}')
                        scp_process = uploaded_worker._ssh.exec_command(f'scp -i {uploaded_worker.SSH_PRIV_KEY} {filepath} {ip}:{filepath}')
                        scp_processes.append(scp_process)

                logging.info(f'Waiting for neural networks be transfered...')
                for sdtin, stdout, stderr in scp_processes:
                    if stdout.channel.recv_exit_status() != 0:
                        logging.error(stderr.read().decode())
                        raise RuntimeError('something wrong happepend during file transfer')
                logging.info(f'Neural network weights uploaded successfully')

    @staticmethod
    def divide_iterations(total_iterations, total_workers):
        worker_total_iterations = [0] * total_iterations
        for i in range(total_iterations):
            worker_total_iterations[i % total_workers] += 1
        return  worker_total_iterations
