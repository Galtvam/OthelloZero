import os
import base64
import pickle
import inspect
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Net.NNet import NNetWrapper

import training

def convert_to_base64_pickle(obj):
    return base64.b64encode(pickle.dumps(obj)).decode('ascii')


def unpack_base64_pickle(pickle_dump):
    return pickle.loads(base64.b64decode(pickle_dump))


def execute_episode(board_size, weights_file, degree_exploration, 
                    num_simulations, policy_temperature, e_greedy):
    neural_network = NNetWrapper(board_size=(board_size, board_size))
    neural_network.load_checkpoint(weights_file)
    
    return training.execute_episode(board_size, neural_network, degree_exploration, 
                                    num_simulations, policy_temperature, e_greedy)

def duel_between_neural_networks(board_size, weights_1_file, weights_2_file, degree_exploration, num_simulations):
    neural_network_1 = NNetWrapper(board_size=(board_size, board_size))
    neural_network_1.load_checkpoint(weights_1_file)

    neural_network_2 = NNetWrapper(board_size=(board_size, board_size))
    neural_network_2.load_checkpoint(weights_2_file)
    
    return training.duel_between_neural_networks(board_size, neural_network_1, neural_network_2, 
                                                 degree_exploration, num_simulations)


def evaluate_neural_network(board_size, total_iterations, weights_1_file, num_simulations, degree_exploration, 
                            agent_class, agent_arguments):
    # FIXME This function doesn't work if agent_class is a NeuralNetworkOthelloAgent,
    # agent_arguments must pass a neural network
    neural_network = NNetWrapper(board_size=(board_size, board_size))
    neural_network.load_checkpoint(weights_file)
    
    return training.evaluate_neural_network(board_size, total_iterations, neural_network, num_simulations, degree_exploration, 
                                            agent_class, agent_arguments)


def pack_arguments_to_pickle(*args):
    return [convert_to_base64_pickle(o) for o in args]


def unpack_pickle_arguments(*args):
    return [unpack_base64_pickle(o) for o in args]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('command', choices=['execute_episode', 
                                            'duel_between_neural_networks', 
                                            'evaluate_neural_network'], help='Training method')
    parser.add_argument('args', nargs='+', help='Method arguments in Base-64 Pickle format')

    args = parser.parse_args()
    
    if args.command == 'execute_episode':
        method = execute_episode
    elif args.command == 'duel_between_neural_networks':
        method = duel_between_neural_networks
    elif args.command == 'evaluate_neural_network':
        method = evaluate_neural_network
    
    parameters = unpack_pickle_arguments(*args.args)
    result = method(*parameters)
    result = convert_to_base64_pickle(result)

    print(result)
