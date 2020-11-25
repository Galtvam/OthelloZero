import random
import logging
import argparse
import numpy as np
import concurrent.futures


from Net.NNet import NNetWrapper, NeuralNets
from MCTS import MCTS, hash_ndarray
from Othello import OthelloGame, OthelloPlayer, BoardView
from Othello.random_agent import *

LOG_FORMAT = '[%(threadName)s] %(asctime)s %(levelname)s: %(message)s'
DEFAULT_CHECKPOINT_FILEPATH = './othelo_model_weights'


class OthelloMCTS(MCTS):
    def __init__(self, board_size, neural_network, degree_exploration):
        self._board_size = board_size
        self._neural_network = neural_network
        self._predict_cache = {}

        if self._neural_network.network_type is NeuralNets.ONN:
            self._board_view_type = BoardView.TWO_CHANNELS
        elif self._neural_network.network_type is NeuralNets.BNN:
            self._board_view_type = BoardView.ONE_CHANNEL

        super().__init__(degree_exploration)
    
    def simulate(self, state, player):
        board = np.copy(state)
        if player is OthelloPlayer.WHITE:
            board = OthelloGame.invert_board(board)
        return super().simulate(board)
    
    def is_terminal_state(self, state):
        return OthelloGame.has_board_finished(state)
    
    def get_state_value(self, state):
        return self._neural_network_predict(state)[1]

    def get_state_reward(self, state):
        return OthelloGame.get_board_winning_player(state)[0].value

    def get_state_actions_propabilities(self, state):
        return self._neural_network_predict(state)[0]
    
    def get_state_actions(self, state):
        return [tuple(a) for a in OthelloGame.get_player_valid_actions(state, OthelloPlayer.BLACK)]
    
    def get_next_state(self, state, action):
        board = np.copy(state)
        OthelloGame.flip_board_squares(board, OthelloPlayer.BLACK, *action)
        if OthelloGame.has_player_actions_on_board(board, OthelloPlayer.WHITE):
            # Invert board to keep using BLACK perspective
            board = OthelloGame.invert_board(board)
        return board

    def get_policy_action_probabilities(self, state, temperature):
        probabilities = np.zeros((self._board_size, self._board_size))

        if temperature == 0:
            for action in self._get_state_actions(state):
                row, col = action
                probabilities[row, col] = self.N(state, action)
            bests = np.argwhere(probabilities == probabilities.max())
            row, col = random.choice(bests)
            probabilities = np.zeros((self._board_size, self._board_size))
            probabilities[row, col] = 1
            return probabilities

        for action in self._get_state_actions(state):
            row, col = action
            probabilities[row, col] = self.N(state, action) ** (1 / temperature)
        return probabilities / (np.sum(probabilities) or 1)

    def moves_scaled_by_valid_moves(self, state):
        network_probabilities = self.get_state_actions_propabilities(state)
        mask = self._mask_valid_moves(state)
        probabilities = network_probabilities * mask
        return probabilities
    
    def _mask_valid_moves(self, state):
        board_mask = np.zeros((self._board_size, self._board_size))
        for action in self._get_state_actions(state):
            row, col = action
            board_mask[row, col] = 1
        return board_mask

    def _neural_network_predict(self, state):
        hash_ = hash_ndarray(state)
        if hash_ not in self._predict_cache:
            if self._board_view_type == BoardView.ONE_CHANNEL:
                state = OthelloGame.convert_to_one_channel_board(state)
            self._predict_cache[hash_] = self._neural_network.predict(state)
        return self._predict_cache[hash_]
    

def execute_episode(board_size, neural_network, degree_exploration, num_simulations, policy_temperature, e_greedy):
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

        #e-greedy
        coin = random.random()
        if coin <= e_greedy:
            action = np.argwhere(policy == policy.max())[0]
        else:
            action = mcts.get_state_actions(state)[np.random.choice(len(mcts.get_state_actions(state)))]


        action_choosed = np.zeros((board_size, board_size))
        action_choosed[action[0]][action[1]] = 1

        #save examples
        if board_view_type == BoardView.ONE_CHANNEL:
            example = game.board(BoardView.ONE_CHANNEL), action_choosed, game.current_player
        else:
            example = state, action_choosed, game.current_player
        examples.append(example)
        
        game.play(*action)

    logging.info(game.board(BoardView.ONE_CHANNEL))
    winner, winner_points = game.get_winning_player()
    logging.info(f'The Winner obtained: {winner_points} points.')

    return [(state, policy, 1 if winner == player else -1) for state, policy, player in examples]


def duel_between_neural_networks(board_size, neural_network_1, neural_network_2, degree_exploration, num_simulations):
    game = OthelloGame(board_size)

    players_neural_networks = {
        OthelloPlayer.BLACK: neural_network_1,
        OthelloPlayer.WHITE: neural_network_2
    }

    neural_networks_mcts = {
      neural_network_1 : OthelloMCTS(board_size, neural_network_1, degree_exploration),
      neural_network_2 : OthelloMCTS(board_size, neural_network_2, degree_exploration)
    }

    while not game.has_finished():
        nn = players_neural_networks[game.current_player]
        state = game.board(BoardView.TWO_CHANNELS)

        logging.info(f'Round: {game.round}')
        for _ in range(num_simulations):
            neural_networks_mcts[nn].simulate(state, game.current_player)

        if game.current_player == OthelloPlayer.WHITE:
            state = OthelloGame.invert_board(state)

        action_probabilities = neural_networks_mcts[nn].get_policy_action_probabilities(state, 0)
        valid_actions = game.get_valid_actions()
        best_action = max(valid_actions, key=lambda position: action_probabilities[tuple(position)])
        game.play(*best_action)

    return players_neural_networks[game.get_winning_player()[0]]


def machine_move(game, neural_networks_mcts, num_simulations):
    '''
    Makes the movement of the machine according to the policy already known
    '''

    state = game.board(BoardView.TWO_CHANNELS)
    for _ in range(num_simulations):
        neural_networks_mcts.simulate(state, game.current_player)

    if game.current_player == OthelloPlayer.WHITE:
        state = OthelloGame.invert_board(state)

    action_probabilities = neural_networks_mcts.get_policy_action_probabilities(state, 0)

    valid_actions = game.get_valid_actions()
    best_action = max(valid_actions, key=lambda position: action_probabilities[tuple(position)])
    game.play(*best_action)


def training(board_size, num_iterations, num_episodes, num_simulations, degree_exploration, temperature,
            total_games, victory_threshold, neural_network, random_agent_interval, random_agent_fights, games_interval,
            e_greedy, temperature_threshold=None, checkpoint_filepath=None, episode_thread_pool=1, 
            game_thread_pool=1, net_type=NeuralNets.ONN):

    total_episodes_done = 0
    historic = []
    training_examples = []
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


        if games_interval > 0 and i % games_interval:
            logging.info(f'Iteration {i}/{num_iterations}: Self-play to evaluate the neural network training')
            
            new_net_victories = 0

            logging.info(f'Iteration {i}/{num_iterations} - Generating matches')

            with concurrent.futures.ThreadPoolExecutor(max_workers=game_thread_pool) as executor:
                future_results = {}
                neural_networks = [old_neural_network, neural_network]            

                for g in range(1, total_games + 1):
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
                        logging.info(f'Iteration {i}/{num_iterations} - Game {g}/{total_games}: New neural network has won')
                        new_net_victories += 1
                    else:
                        logging.info(f'Iteration {i}/{num_iterations} - Game {g}/{total_games}: New neural network has lost')

                    logging.info(f'Iteration {i}/{num_iterations} - Game {g}/{total_games}: ' 
                                f'Promotion status ({new_net_victories}/{victory_threshold})')

            if new_net_victories >= victory_threshold:
                logging.info(f'Iteration {i}/{num_iterations}: New neural network has been promoted')
                
                neural_network.save_checkpoint(checkpoint_filepath)
                logging.info(f'Iteration {i}/{num_iterations}: Saving trained model in "{checkpoint_filepath}"')
            else:
                neural_network = old_neural_network
        
        # gambiarra
        neural_network.save_checkpoint(checkpoint_filepath)

        if (i % random_agent_interval) == 0:
            color = [OthelloPlayer.BLACK, OthelloPlayer.WHITE]
            net_wins = 0

            for k in range(random_agent_fights):
                random.shuffle(color)
                game = OthelloGame(board_size, current_player = OthelloPlayer.BLACK)
                neural_networks_mcts = OthelloMCTS(board_size, neural_network, degree_exploration=1)

                while not game.has_finished():
                    #neural network move
                    if game.current_player is color[0]:
                        machine_move(game, neural_networks_mcts, num_simulations)

                    #random agent move
                    else:
                        random_agent(game)
                
                winner, points = game.get_winning_player()       
                logging.info(f'The player {winner} won with {points} points')
                
                if winner == color[0]:
                    net_wins += 1
                    logging.info(f'Total Episodes Runned: {total_episodes_done} - Network won: {net_wins}/{k+1}')
                else:
                    logging.info(f'Total Episodes Runned: {total_episodes_done} - Network lost: {net_wins}/{k+1}')
            
            historic.append( (total_episodes_done, (net_wins/random_agent_fights)) )
            logging.info(historic)

            with open(f'historic-last-training-session-{board_size}.txt', 'w') as output:
                output.write(str(historic))
            with open(f'examples-{board_size}.txt', 'w') as output:
                output.write(str(training_examples))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--board-size', default=6, type=int, help='Othello board size')
    parser.add_argument('-i', '--iterations', default=80, type=int, help='Number of training iterations')
    parser.add_argument('-e', '--episodes', default=100, type=int, help='Number of episodes by iterations')
    parser.add_argument('-s', '--simulations', default=25, type=int, help='Number of MCTS simulations by episode')
    parser.add_argument('-g', '--total-games', default=10, type=int, help='Total of games to evaluate neural network training')
    parser.add_argument('-v', '--victory-threshold', default=6, type=int, help='Number of victories to promote neural network training')
    parser.add_argument('-c', '--constant-upper-confidence', default=1, type=int, help='MCTS upper confidence bound constant')
    parser.add_argument('-eg', '--e-greedy', default=0.9, type=float, help='e constant used in e-greedy')

    parser.add_argument('-n', '--network-type', default=1, type=int, help='1- OthelloNN, 2- BaseNN')

    parser.add_argument('-ri', '--random-agent-interval', default=5, type=int, help='Number of iterations between evaluation against random agent')
    parser.add_argument('-rf', '--random-agent-fights', default=10, type=int, help='Number of fights against random agent')

    parser.add_argument('-gi', '--games-interval', default=1, type=int, help='Number of iterations between self-play games')

    parser.add_argument('-ep', '--epochs', default=10, type=int, help='Number of epochs for neural network training')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help='Neural network training learning rate')
    parser.add_argument('-dp', '--dropout', default=0.3, type=float, help='Neural network training dropout')
    parser.add_argument('-bs', '--batch-size', default=32, type=int, help='Neural network training batch size')
    
    parser.add_argument('-et', '--episode-threads', default=1, type=int, help='Number of episodes to be executed asynchronously')
    parser.add_argument('-gt', '--game-threads', default=1, type=int, help='Number of games to be executed asynchronously '
                                                                           'during evaluation')

    parser.add_argument('-o', '--output-file', default=DEFAULT_CHECKPOINT_FILEPATH, help='File path to save neural network weights')
    parser.add_argument('-w', '--weights-file', default=None, help='File path to load neural network weights')
    parser.add_argument('-l', '--log-level', default='DEBUG', choices=('INFO', 'DEBUG', 'WARNING', 'ERROR'), help='Logging level')
    parser.add_argument('-t', '--temperature', default=1, type=int, help='Policy temperature parameter')
    parser.add_argument('-tt', '--temperature-threshold', default=25, type=int, help='Number of iterations using the temperature '
                                                                                     'parameter before changing to 0')
    
    args = parser.parse_args()

    assert args.victory_threshold <= args.total_games, '"victory-threshold" must be less than "total-games"'

    logging.basicConfig(level=getattr(logging, args.log_level, None), format=LOG_FORMAT)

    net_type = NeuralNets.ONN if args.network_type == 1 else NeuralNets.BNN
    
    neural_network = NNetWrapper(board_size=(args.board_size, args.board_size), batch_size=args.batch_size,
                                 epochs=args.epochs, lr=args.learning_rate, dropout=args.dropout, network=net_type)
    if args.weights_file:
        neural_network.load_checkpoint(args.weights_file)

    training(board_size=args.board_size, num_iterations=args.iterations, num_episodes=args.episodes, num_simulations=args.simulations, 
             degree_exploration=args.constant_upper_confidence, temperature=args.temperature,  total_games=args.total_games, 
             victory_threshold=args.victory_threshold, neural_network=neural_network, temperature_threshold=args.temperature_threshold, 
             checkpoint_filepath=args.output_file, episode_thread_pool=args.episode_threads, game_thread_pool=args.game_threads, 
             net_type=args.network_type, random_agent_interval=args.random_agent_interval, random_agent_fights=args.random_agent_fights, 
             games_interval=args.games_interval, e_greedy=args.e_greedy)
