import os
import random
import logging
import argparse
import numpy as np
import concurrent.futures


from othelo_mcts import OthelloMCTS
from agents import RandomOthelloAgent, NeuralNetworkOthelloAgent, duel_between_agents

from Othello import OthelloGame, OthelloPlayer, BoardView

LOG_FORMAT = '[%(threadName)s] %(asctime)s %(levelname)s: %(message)s'
DEFAULT_CHECKPOINT_FILEPATH = './othelo_model_weights'


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
    logging.info(f'Episode: The winner obtained: {winner_points} points.')

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

    return agents[agent_winner]


def evaluate_neural_network(total_iterations, neural_network, degree_exploration, agent_class, agent_arguments):
    net_wins = 0
    board_size = neural_network.board_size_x
    color = [OthelloPlayer.BLACK, OthelloPlayer.WHITE]

    for iteration in range(1, total_iterations + 1):
        random.shuffle(color)
        game = OthelloGame(board_size)
        neural_networks_mcts = OthelloMCTS(board_size, neural_network, degree_exploration)
        agent = agent_class(game, **agent_arguments)

        while not game.has_finished():
            if game.current_player is color[0]:
                agent.play()
        
        winner, points = game.get_winning_player()       
        logging.info(f'The player {winner} won with {points} points')
        
        if winner == color[0]:
            net_wins += 1
            logging.info(f'Neural Network Evaluation: Network won - {net_wins}/{iteration}')
        else:
            logging.info(f'Neural Network Evaluation: Network lost - {net_wins}/{iteration}')
    
    return total_episodes_done, (net_wins / random_agent_fights)

    with open(f'historic-last-training-session-{board_size}.txt', 'w') as output:
        output.write(str(historic))


def training(board_size, num_iterations, num_episodes, num_simulations, degree_exploration, temperature, neural_network, 
             e_greedy, evaluation_interval, evaluation_iterations, evaluation_agent_class, evaluation_agent_arguments, 
             temperature_threshold, self_play_training, self_play_interval, self_play_total_games, 
             self_play_threshold, episode_thread_pool, game_thread_pool, checkpoint_filepath):
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=episode_thread_pool) as executor:
            future_results = {}
            
            for e in range(1, num_episodes + 1):
                total_episodes_done += 1
                future_result = executor.submit(execute_episode, board_size, neural_network, degree_exploration, 
                                                num_simulations, temperature, e_greedy)
                future_results[future_result] = e

            logging.info(f'Iteration {i}/{num_iterations} - Waiting for episodes results')

            for future in concurrent.futures.as_completed(future_results):
                e = future_results[future]
                logging.info(f'Iteration {i}/{num_iterations} - Episode {e}: Finished')
                episode_examples = future.result()
                training_examples.extend(episode_examples)

        logging.info(f'Iteration {i}/{num_iterations}: All episodes finished')
        
        training_verbose = 2 if logging.root.level <= logging.DEBUG else None

        logging.info(f'Iteration {i}/{num_iterations}: Training model with episodes examples')

        random.shuffle(training_examples)
        history = neural_network.train(training_examples, verbose=training_verbose)

        if self_play_training and i % self_play_interval == 0:
            logging.info(f'Iteration {i}/{num_iterations}: Self-play to evaluate the neural network training')
            
            new_net_victories = 0

            logging.info(f'Iteration {i}/{num_iterations} - Generating matches')

            with concurrent.futures.ThreadPoolExecutor(max_workers=game_thread_pool) as executor:
                future_results = {}
                neural_networks = [old_neural_network, neural_network]            

                for g in range(1, self_play_total_games + 1):
                    random.shuffle(neural_networks)
                    future_result = executor.submit(duel_between_neural_networks, board_size, 
                                                    neural_networks[0], neural_networks[1],
                                                    degree_exploration, num_simulations)
                    future_results[future_result] = g

                logging.info(f'Iteration {i}/{num_iterations} - Waiting for matches results')
            
                for future in concurrent.futures.as_completed(future_results):
                    g = future_results[future]
                    winner = future.result()
                    if winner is neural_network:
                        logging.info(f'Iteration {i}/{num_iterations} - Game {g}/{self_play_total_games}: New neural network has won')
                        new_net_victories += 1
                    else:
                        logging.info(f'Iteration {i}/{num_iterations} - Game {g}/{self_play_total_games}: New neural network has lost')

                    logging.info(f'Iteration {i}/{num_iterations} - Game {g}/{self_play_total_games}: ' 
                                f'Promotion status ({new_net_victories}/{victory_threshold})')

            if new_net_victories >= self_play_threshold:
                logging.info(f'Iteration {i}/{num_iterations}: New neural network has been promoted')
                
                neural_network.save_checkpoint(checkpoint_filepath)
                logging.info(f'Iteration {i}/{num_iterations}: Saving trained model in "{checkpoint_filepath}"')
            else:
                neural_network = old_neural_network
        else:
            neural_network.save_checkpoint(checkpoint_filepath)

        if i % evaluation_interval == 0:
            logging.info(f'Evaluating Neural Network')
            evaluation_result = evaluate_neural_network(evaluation_iterations, neural_network, degree_exploration, 
                                                        evaluation_agent_class, evaluation_agent_arguments)
            logging.info(f'Neural Network Evaluation final result: {evaluation_result}')
            historic.append(result)

        with open(f'examples-{board_size}.txt', 'w') as output:
            output.write(str(training_examples))


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
    
    parser.add_argument('-ea', '--evaluation-agent', default='random', choices=('random'), help='Agent for neural network evaluation')
    parser.add_argument('-ei', '--evaluation-interval', default=5, type=int, help='Number of iterations between evaluations')
    parser.add_argument('-eg', '--evaluation-games', default=10, type=int, help='Number of matches against the evaluation agent')

    parser.add_argument('-ep', '--epochs', default=10, type=int, help='Number of epochs for neural network training')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help='Neural network training learning rate')
    parser.add_argument('-dp', '--dropout', default=0.3, type=float, help='Neural network training dropout')
    parser.add_argument('-bs', '--batch-size', default=32, type=int, help='Neural network training batch size')
    
    parser.add_argument('-et', '--episode-threads', default=1, type=int, help='Number of episodes to be executed asynchronously')
    parser.add_argument('-gt', '--game-threads', default=1, type=int, help='Number of games to be executed asynchronously '
                                                                           'during evaluation')

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

    from Net.NNet import NNetWrapper, NeuralNets

    if args.self_play:
        assert args.self_play_threshold <= args.self_play_games, '"self-play-threshold" must be less than "self-play-games"'

    logging.basicConfig(level=getattr(logging, args.log_level, None), format=LOG_FORMAT)

    net_type = NeuralNets.ONN if args.network_type == 1 else NeuralNets.BNN
    
    neural_network = NNetWrapper(board_size=(args.board_size, args.board_size), batch_size=args.batch_size,
                                 epochs=args.epochs, lr=args.learning_rate, dropout=args.dropout, network=net_type)
    if args.weights_file:
        neural_network.load_checkpoint(args.weights_file)

    evaluation_agent_class = RandomOthelloAgent
    evaluation_agent_arguments = dict()

    training(board_size=args.board_size, num_iterations=args.iterations, num_episodes=args.episodes, num_simulations=args.simulations, 
             degree_exploration=args.constant_upper_confidence, temperature=args.temperature, neural_network=neural_network, 
             e_greedy=args.e_greedy, evaluation_interval=args.evaluation_interval, evaluation_iterations=args.evaluation_games, 
             evaluation_agent_class=evaluation_agent_class, evaluation_agent_arguments=evaluation_agent_arguments,
             temperature_threshold=args.temperature_threshold, self_play_training=args.self_play, 
             self_play_interval=args.self_play_interval, self_play_total_games=args.self_play_games, 
             self_play_threshold=args.self_play_threshold, episode_thread_pool=args.episode_threads, 
             game_thread_pool=args.game_threads, checkpoint_filepath=args.output_file)
