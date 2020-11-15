import argparse
import random
import numpy as np

from Net.NNet import NNetWrapper
from MCTS import MCTS, hash_ndarray
from Othello import OthelloGame, OthelloPlayer, BoardView


class OthelloMCTS(MCTS):
    def __init__(self, board_size, neural_network, degree_exploration):
        self._board_size = board_size
        self._neural_network = neural_network
        self._predict_cache = {}

        super().__init__(degree_exploration)
    
    def is_terminal_state(self, state):
        return OthelloGame.has_board_finished(state)
    
    def get_state_value(self, state):
        return self._neural_network_predict(state)[1]

    def get_state_reward(self, state):
        return OthelloGame.get_board_winning_player(state)[0].value

    def get_state_actions_propabilities(self, state):
        return self._neural_network_predict(state)
    
    def get_state_actions(self, state):
        return [tuple(a) for a in OthelloGame.get_player_valid_actions(state, OthelloPlayer.BLACK)]
    
    def get_next_state(self, state, action):
        board = np.copy(state)
        next_state = OthelloGame.flip_board_squares(board, OthelloPlayer.BLACK, *action)

        if OthelloGame.has_player_actions_on_board(board, OthelloPlayer.WHITE):
            # Invert board to keep using BLACK perspective
            return OthelloGame.invert_board(board)
        return board

    def get_policy_action_probabilities(self, state, temperature):
        probabilities = np.zeros((self._board_size, self._board_size))

        if temperature == 0:
            for action in self._get_state_actions(state):
                row, col = action
                probabilities[row, col] = self.N(state, action)
            bests = np.argwhere(probabilities[row, col] == probabilities.max())
            row, col = random.choice(bests)
            probabilities = np.zeros((self._board_size, self.board_size))
            probabilities[row, col] = 1
            return probabilities

        for action in self._get_state_actions(state):
            row, col = action
            probabilities[row, col] = self.N(state, action) ** (1 / temperature)
        
        return probabilities / np.sum(probabilities)

    def _neural_network_predict(self, state):
        hash_ = hash_ndarray(state)
        if hash_ not in self._predict_cache:
            self._predict_cache[hash_] = self._neural_network.predict(state)
        return self._predict_cache[hash_]


def execute_episode(board_size, neural_network, degree_exploration, num_simulations, policy_temperature):
    examples = []
    
    game = OthelloGame(board_size)

    mcts = OthelloMCTS(board_size, neural_network, degree_exploration)

    while not game.has_finished(): 
        state = game.board(BoardView.TWO_CHANNELS)
        for _ in range(num_simulations):
            mcts.simulate(state)

        policy = mcts.get_policy_action_probabilities(state, policy_temperature)
        
        if game.current_player == OthelloPlayer.WHITE:
            state = OthelloGame.invert_board(state)

        example = state, policy, game.current_player
        examples.append(example)

        action = np.argwhere(policy == policy.max())[0]
        
        game.play(*action)

    winner, winner_points = game.get_winning_player()

    return [(state, policy, 1 if winner == player else -1) for state, policy, player in examples]


def duel_between_neural_networks(board_size, neural_network_1, neural_network_2):
    game = OthelloGame(board_size)

    players_neural_networks = {
        OthelloPlayer.BLACK: neural_network_1,
        OthelloPlayer.WHITE: neural_network_2
    }

    while not game.has_finished():
        nn = players_neural_networks[game.current_player]
        action_probabilities, state_value = nn.predict(game.board(BoardView.TWO_CHANNELS))
        valid_actions = game.get_valid_actions()
        best_action = max(valid_actions, key=lambda position: action_probabilities[tuple(position)])
        game.play(*best_action)
        print(game.board())

    return game.get_winning_player()[0]    


if __name__ == '__main__':
    nn = NNetWrapper((8, 8))
    nn2 = NNetWrapper((8, 8))
    print(duel_between_neural_networks(8, nn, nn2))
