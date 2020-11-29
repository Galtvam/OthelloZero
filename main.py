import os
import random
import logging
import argparse

import googleapiclient.discovery
import gcloud

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

LOG_FORMAT = '[%(threadName)s] %(asctime)s %(levelname)s: %(message)s'
DEFAULT_CHECKPOINT_FILEPATH = './othelo_model_weights.h5'



def training(board_size, num_iterations, num_episodes, num_simulations, degree_exploration, temperature, neural_network, 
             e_greedy, evaluation_interval, evaluation_iterations, evaluation_agent_class, evaluation_agent_arguments, 
             temperature_threshold, self_play_training, self_play_interval, self_play_total_games, 
             self_play_threshold, checkpoint_filepath, worker_manager):
    
    if self_play_training:
        assert self_play_threshold <= self_play_total_games, 'Self-play threshold must be less than self-play games'

    assert evaluation_iterations % worker_manager.total_workers() == 0, \
        'Evaluation iterations must be divisible equally between the workers'

    historic = []
    training_examples = []
    total_episodes_done = 0
    for i in range(1, num_iterations + 1):
        old_neural_network = neural_network.copy()

        logging.info(f'Iteration {i}/{num_iterations}: Starting iteration')
        
        if temperature_threshold and i >= temperature_threshold:
            logging.info(f'Iteration {i}/{num_iterations}: Temperature threshold reached, '
                          'changing temperature to 0')
            temperature = 0

        logging.info(f'Iteration {i}/{num_iterations} - Generating episodes')

        logging.info(f'Iteration {i}/{num_iterations} - Waiting for episodes results')
        # Does not matter if its we execute episode on old_neural_network or neural_network
        # To avoid bugs with GoogleCloudWorker cache use old_neural_network
        # Changing this, the worker consider will neural_network cache instead of
        # using the trained neural_network
        worker_manager.run(WorkType.EXECUTE_EPISODE, num_episodes, board_size, 
                           old_neural_network, degree_exploration, num_simulations, 
                           temperature, e_greedy)
        
        total_episodes_done += num_episodes

        for training_example in worker_manager.get_results():
            training_examples.extend(training_example)

        logging.info(f'Iteration {i}/{num_iterations}: All episodes finished')
        
        training_verbose = 2 if logging.root.level <= logging.DEBUG else None

        logging.info(f'Iteration {i}/{num_iterations}: Training model with episodes examples')

        random.shuffle(training_examples)
        
        history = neural_network.train(training_examples, verbose=training_verbose)

        if self_play_training and i % self_play_interval == 0:
            logging.info(f'Iteration {i}/{num_iterations}: Self-play to evaluate the neural network training')
            
            self_play_results = []

            logging.info(f'Iteration {i}/{num_iterations} - Generating self-play matches')

            logging.info(f'Iteration {i}/{num_iterations} - Waiting for self-play matches results')
            worker_manager.run(WorkType.DUEL_BETWEEN_NEURAL_NETWORKS, self_play_total_games, 
                               board_size, neural_network, old_neural_network, 
                               degree_exploration, num_simulations)
            
            for winner in worker_manager.get_results():
                if winner == 0:
                    self_play_results.append(neural_network)
                else:
                    self_play_results.append(old_neural_network)

            # logging.info(f'Iteration {i}/{num_iterations} - Generating BLACK x WHITE matches')

            # logging.info(f'Iteration {i}/{num_iterations} - Waiting for BLACK x WHITE matches results')
            # worker_manager.run(WorkType.DUEL_BETWEEN_NEURAL_NETWORKS, self_play_total_games // 2, 
            #                    board_size, neural_network, old_neural_network, 
            #                    degree_exploration, num_simulations)
            
            # for winner in worker_manager.get_results():
            #     if winner == 0:
            #         self_play_results.append(neural_network)
            #     else:
            #         self_play_results.append(old_neural_network)

            # logging.info(f'Iteration {i}/{num_iterations} - Generating WHITE x BLACK matches')

            # logging.info(f'Iteration {i}/{num_iterations} - Waiting for WHITE x BLACK matches results')
            # worker_manager.run(WorkType.DUEL_BETWEEN_NEURAL_NETWORKS, (self_play_total_games // 2) + self_play_total_games % 2, 
            #                    board_size, old_neural_network, neural_network, 
            #                    degree_exploration, num_simulations)
            
            # for winner in worker_manager.get_results():
            #     if winner == 0:
            #         self_play_results.append(old_neural_network)
            #     else:
            #         self_play_results.append(neural_network)

            new_net_victories = len([1 for winner in self_play_results if winner is neural_network])

            logging.info(f'Iteration {i}/{num_iterations} - Game results: {new_net_victories}/{self_play_total_games}: ')

            if new_net_victories >= self_play_threshold:
                logging.info(f'Iteration {i}/{num_iterations}: New neural network has been promoted')
                
                neural_network.save_checkpoint(checkpoint_filepath)
                logging.info(f'Iteration {i}/{num_iterations}: Saving trained model in "{checkpoint_filepath}"')
            else:
                neural_network = old_neural_network
                logging.info(f'Iteration {i}/{num_iterations}: New neural network has not been promoted')
        else:
            neural_network.save_checkpoint(checkpoint_filepath)

        if i % evaluation_interval == 0:
            logging.info(f'Evaluating Neural Network')
            logging.info(f'Iteration {i}/{num_iterations} - Waiting for evaluation results')
            worker_manager.run(WorkType.EVALUATE_NEURAL_NETWORK, worker_manager.total_workers(), 
                               board_size, evaluation_iterations // worker_manager.total_workers(), neural_network, num_simulations,
                               degree_exploration, evaluation_agent_class, evaluation_agent_arguments)
            
            neural_network_wins = sum(worker_manager.get_results())

            evaluation_result = neural_network_wins / evaluation_iterations
            logging.info(f'Neural Network Evaluation final result: {evaluation_result}')
            historic.append(evaluation_result)
        
        logging.info(f'Total episodes done: {total_episodes_done}')

        with open(f'examples-{board_size}.txt', 'w') as output:
            output.write(str(training_examples))

        with open(f'historic-last-training-session-{board_size}.txt', 'w') as output:
            output.write(str(historic))
        
        worker_manager.clear_cache()
    
    return historic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--board-size', default=6, type=int, help='Othello board size')
    parser.add_argument('-i', '--iterations', default=80, type=int, help='Number of training iterations')
    parser.add_argument('-e', '--episodes', default=100, type=int, help='Number of episodes by iterations')
    parser.add_argument('-s', '--simulations', default=25, type=int, help='Number of MCTS simulations by episode')
    parser.add_argument('-c', '--constant-upper-confidence', default=1, type=int, help='MCTS upper confidence bound constant')
    parser.add_argument('-g', '--e-greedy', default=0.9, type=float, help='e constant used in e-greedy')

    parser.add_argument('-n', '--network-type', default=1, choices=(1, 2), help='1- OthelloNN, 2- BaseNN')

    parser.add_argument('-sp', '--self-play', default=False, action='store_true', help='Do self-play at end of each iteration')
    parser.add_argument('-si', '--self-play-interval', default=1, type=int, help='Number of iterations between self-play games')
    parser.add_argument('-sg', '--self-play-games', default=10, type=int, help='Number of games during self-play games')
    parser.add_argument('-st', '--self-play-threshold', default=6, type=int, help='Number of victories to promote neural network')

    parser.add_argument('-ea', '--evaluation-agent', default='random', choices=['random'], help='Agent for neural network evaluation')
    parser.add_argument('-ei', '--evaluation-interval', default=5, type=int, help='Number of iterations between evaluations')
    parser.add_argument('-eg', '--evaluation-games', default=12, type=int, help='Number of matches against the evaluation agent')

    parser.add_argument('-ep', '--epochs', default=10, type=int, help='Number of epochs for neural network training')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help='Neural network training learning rate')
    parser.add_argument('-dp', '--dropout', default=0.3, type=float, help='Neural network training dropout')
    parser.add_argument('-bs', '--batch-size', default=32, type=int, help='Neural network training batch size')

    parser.add_argument('-tw', '--thread-workers', default=1, type=int, help='Number of Thread workers to do training tasks')
    parser.add_argument('-gw', '--google-workers', default=False, action='store_true', help='Use Google Cloud workers')
    parser.add_argument('-gt', '--google-workers-label', default=gcloud.INSTANCE_LABEL[0],
                        help='Tag of Google Cloud machines which will be as worker')
    parser.add_argument('-gc', '--google-credentials', default=None, 
                        help='Google Cloud API Credentials JSON file path')
    parser.add_argument('-gp', '--google-project', default=None, help='Google Cloud Platform project name')
    parser.add_argument('-gz', '--google-zone', default=gcloud.DEFAULT_ZONE, 
                        help='Google Cloud Platform instances zone')
    parser.add_argument('-gk', '--google-key-filename', default=None, 
                        help='Google Cloud SSH Private key')

    parser.add_argument('-o', '--output-file', default=DEFAULT_CHECKPOINT_FILEPATH, help='File path to save neural network weights')
    parser.add_argument('-w', '--weights-file', default=None, help='File path to load neural network weights')
    parser.add_argument('-l', '--log-level', default='INFO', choices=('INFO', 'DEBUG', 'WARNING', 'ERROR'), help='Logging level')
    parser.add_argument('-t', '--temperature', default=1, type=int, help='Policy temperature parameter')
    parser.add_argument('-tt', '--temperature-threshold', default=25, type=int, help='Number of iterations using the temperature '
                                                                                        'parameter before changing to 0')

    parser.add_argument('-ug', '--use-gpu', default=False, action='store_true', help='Enable GPU for Tensorflow')

    args = parser.parse_args()

    if not args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from agents import RandomOthelloAgent
    from Net.NNet import NNetWrapper, NeuralNets
    from workers import WorkerManager, WorkType, ThreadWorker, GoogleCloudWorker

    logging.basicConfig(level=getattr(logging, args.log_level, None), format=LOG_FORMAT)

    net_type = NeuralNets.ONN if args.network_type == 1 else NeuralNets.BNN

    neural_network = NNetWrapper(board_size=(args.board_size, args.board_size), batch_size=args.batch_size,
                                    epochs=args.epochs, lr=args.learning_rate, dropout=args.dropout, network=net_type)
    if args.weights_file:
        neural_network.load_checkpoint(args.weights_file)

    evaluation_agent_class = RandomOthelloAgent
    evaluation_agent_arguments = dict()

    worker_manager = WorkerManager()
    worker_manager.add_worker(ThreadWorker())
    
    if args.google_workers:
        assert args.google_credentials, 'Google Cloud Credentials required'
        assert args.google_project, 'Google Cloud Project name required'
        assert args.google_zone, 'Google Cloud instances zone required'
        assert args.google_key_filename, 'Google Cloud SSH Private key required'
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.google_credentials

        compute = googleapiclient.discovery.build('compute', 'v1')

        instances = gcloud.search_instances(compute, args.google_project, args.google_zone, 
                                            args.google_workers_label, 'true')

        for instance in instances:
            worker = GoogleCloudWorker(compute, args.google_project, args.google_zone, 
                                       instance['name'], args.google_key_filename)
            worker_manager.add_worker(worker)
    else:
        for _ in range(args.workers - 1):
            worker_manager.add_worker(ThreadWorker())

    training(board_size=args.board_size, num_iterations=args.iterations, num_episodes=args.episodes, num_simulations=args.simulations, 
            degree_exploration=args.constant_upper_confidence, temperature=args.temperature, neural_network=neural_network, 
            e_greedy=args.e_greedy, evaluation_interval=args.evaluation_interval, evaluation_iterations=args.evaluation_games, 
            evaluation_agent_class=evaluation_agent_class, evaluation_agent_arguments=evaluation_agent_arguments,
            temperature_threshold=args.temperature_threshold, self_play_training=args.self_play, 
            self_play_interval=args.self_play_interval, self_play_total_games=args.self_play_games, 
            self_play_threshold=args.self_play_threshold, worker_manager=worker_manager, 
            checkpoint_filepath=args.output_file)
