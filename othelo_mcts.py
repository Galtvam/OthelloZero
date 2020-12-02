import random
import numpy as np

from Net.NNet import NeuralNets
from MCTS import MCTS, hash_ndarray
from Othello import BoardView, OthelloPlayer, OthelloGame


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
