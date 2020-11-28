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


def execute_episode(weights_file, board_size, degree_exploration, 
                    num_simulations, policy_temperature, e_greedy):
    neural_network = NNetWrapper(board_size=(board_size, board_size))
    neural_network.load_checkpoint(weights_file)
    
    return training.execute_episode(board_size, neural_network, degree_exploration, 
                                    num_simulations, policy_temperature, e_greedy)

def pack_arguments_to_pickle(*args):
    return [convert_to_base64_pickle(o) for o in args]


def unpack_pickle_arguments(*args):
    return [unpack_base64_pickle(o) for o in args]


if __name__ == "__main__":
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # neural_network = NNetWrapper(board_size=(6, 6))
    # # weights = neural_network.nnet.model.get_weights()
    # # result = convert_to_base64_pickle(weights)
    # _, file = tempfile.mkstemp()
    # neural_network.save_checkpoint(file)
    # print(file)
    # exit()
    # os.remove()
    # # print(result.decode('ascii'))
    # board_size = 6
    # degree_exploration = 1
    # num_simulations = 20
    # policy_temperature = 1
    # e_greedy = 0.25
    # print(convert_to_base64_pickle(board_size))
    # print(convert_to_base64_pickle(degree_exploration))
    # print(convert_to_base64_pickle(num_simulations))
    # print(convert_to_base64_pickle(policy_temperature))
    # print(convert_to_base64_pickle(e_greedy))
    # # board_size, neural_network, degree_exploration, 
    # #                num_simulations, policy_temperature, e_greedy
    # exit()

    # print(pack_arguments_to_pickle('othelo_model_weights', 6, 1, 25, 1, 0.25))

    # exit()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('command', choices=['execute_episode', 
                                            'duel_between_neural_networks', 
                                            'evaluate_neural_network'], help='Training method')
    parser.add_argument('args', nargs='+', help='Method arguments in Base-64 Pickle format')

    args = parser.parse_args()
    
    if args.command == 'execute_episode':
        method = execute_episode
    
    parameters = unpack_pickle_arguments(*args.args)
    result = method(*parameters)
    result = convert_to_base64_pickle(result)

    print(result)
