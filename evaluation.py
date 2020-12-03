import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from Othello import OthelloGame
from Net.NNet import NNetWrapper, NeuralNets
from agents import RandomOthelloAgent, GreedyOthelloAgent, NeuralNetworkOthelloAgent, \
    duel_between_agents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('board_size', type=int, help='Othello board size')
    parser.add_argument('neural_network_weights',
                        help='File path to load neural network weights')
    parser.add_argument('opponent', nargs='?', choices=['random', 'greedy'], default='random', 
                        help='Kind of Othello Opponent')
    parser.add_argument('-g', '--games', default=1, type=int, help='Number of games to be played')
    parser.add_argument('-s', '--simulations', default=35, type=int, help='Number of MCTS simulations')
    
    args = parser.parse_args()

    neural_network = NNetWrapper(board_size=(args.board_size, args.board_size))
    neural_network.load_checkpoint(args.neural_network_weights)

    wins = 0

    for g in range(1, args.games + 1):
        print(f'Game Iteration {g}/{args.games}: Starting...')
        game = OthelloGame(args.board_size)
        nn_agent = NeuralNetworkOthelloAgent(game, neural_network, args.simulations, degree_exploration=1)
        opponent_agent = RandomOthelloAgent(game) if args.opponent == 'random' else GreedyOthelloAgent(game)

        agents = [nn_agent, opponent_agent]
        agent_winner, points = duel_between_agents(game, *agents)

        if nn_agent is agent_winner:
            print(f'Game Iteration {g}/{args.games}: Neural Network won with {points} points')
            wins += 1
        else:
            print(f'Game Iteration {g}/{args.games}: Neural Network lost the game')
        
        print(f'Evaluation status - Win rate: {wins / args.games} ({wins}/{args.games})')
