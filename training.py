import random
import logging
import numpy as np

from Net.NNet import NeuralNets

from othelo_mcts import OthelloMCTS
from agents import NeuralNetworkOthelloAgent, duel_between_agents

from Othello import OthelloGame, OthelloPlayer, BoardView


def execute_episode(board_size, neural_network, degree_exploration, 
                    num_simulations, policy_temperature, e_greedy):
    examples = []
    
    game = OthelloGame(board_size)

    mcts = OthelloMCTS(board_size, neural_network, degree_exploration)

    if neural_network.network_type == NeuralNets.ONN:
        board_view_type = BoardView.TWO_CHANNELS
    elif neural_network.network_type == NeuralNets.BNN:
        board_view_type = BoardView.ONE_CHANNEL

    while not game.has_finished():
        state = game.board(BoardView.TWO_CHANNELS)
        
        for _ in range(num_simulations):
            mcts.simulate(state, game.current_player)

        if game.current_player == OthelloPlayer.WHITE:
            state = OthelloGame.invert_board(state)

        policy = mcts.get_policy_action_probabilities(state, policy_temperature)

        # e-greedy
        coin = random.random()

        if coin <= e_greedy:
            action = np.argwhere(policy == policy.max())[0]
        else:
            action = mcts.get_state_actions(state)[np.random.choice(len(mcts.get_state_actions(state)))]

        action_choosed = np.zeros((board_size, board_size))
        action_choosed[action[0]][action[1]] = 1

        example = game.board(board_view_type), action_choosed, game.current_player
        examples.append(example)
        
        game.play(*action)

    winner, winner_points = game.get_winning_player()
    logging.info(f'Episode finished: The winner obtained {winner_points} points.')

    return [(state, policy, 1 if winner == player else -1) for state, policy, player in examples]


def duel_between_neural_networks(board_size, neural_network_1, neural_network_2, degree_exploration, num_simulations):
    game = OthelloGame(board_size)

    nn_1_agent = NeuralNetworkOthelloAgent(game, neural_network_1, num_simulations, degree_exploration)
    nn_2_agent = NeuralNetworkOthelloAgent(game, neural_network_2, num_simulations, degree_exploration)

    agents = {
        nn_1_agent: neural_network_1,
        nn_2_agent: neural_network_2
    }

    agent_winner = duel_between_agents(game, nn_1_agent, nn_2_agent)

    return 0 if agents[agent_winner] is neural_network_1 else 1


def evaluate_neural_network(board_size, total_iterations, neural_network, num_simulations, degree_exploration, 
                            agent_class, agent_arguments):
    net_wins = 0

    logging.info(f'Neural Network Evaluation: Started')

    for iteration in range(1, total_iterations + 1):

        game = OthelloGame(board_size)

        nn_agent = NeuralNetworkOthelloAgent(game, neural_network, num_simulations, degree_exploration)
        evaluation_agent = agent_class(game, *agent_arguments)

        agents = [evaluation_agent, nn_agent]
        random.shuffle(agents)

        agent_winner = duel_between_agents(game, *agents)
        
        if agent_winner is nn_agent:
            net_wins += 1
            logging.info(f'Neural Network Evaluation: Network won')
        else:
            logging.info(f'Neural Network Evaluation: Network lost')
    
    return net_wins
