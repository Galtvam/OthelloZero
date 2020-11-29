import math
import hashlib
import numpy as np
import logging

import numpy as np

from enum import Enum, auto


class BoardView(Enum):
    ONE_CHANNEL = auto()
    TWO_CHANNELS = auto()


# Representation of the player in one-channel board view
class OthelloPlayer(Enum):
    BLACK = 1
    WHITE = -1
    
    @property
    def opponent(self):
        return OthelloPlayer.WHITE if self is OthelloPlayer.BLACK else OthelloPlayer.BLACK 


class OthelloGame:
    PLAYER_CHANNELS = {
        OthelloPlayer.BLACK: 0,
        OthelloPlayer.WHITE: 1,
    }

    ALL_DIRECTIONS = np.array([(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)])

    def __init__(self, board_size=8, initial_board=None, current_player=OthelloPlayer.BLACK):
        """Create Othello board game representation

        Args:
            board_size ([int]): Size of the board square (4, 6, 8)
            initial_board (ndrray(board_size, board_size, 2)): Initial board
        """
        assert board_size % 2 == 0, 'Board size must be even'

        assert initial_board is None or initial_board.shape == (board_size, board_size, 2), \
            f'Expecting initial board shape ({board_size}, {board_size}, 2)'

        if initial_board is not None:
            self._board = initial_board
        else:
            self._board = self.initial_board(board_size)
        self._board_size = board_size
        self._round = 1

        self.current_player = current_player

        # Save the last time when the board was converted to one channel view,
        # this is to avoid matrix operations when that version of the board has
        # been calculated already
        self._one_channel_board_last_update = None
        self._one_channel_board = None

        if initial_board is not None:
            self._has_finished = OthelloGame.has_board_finished(self._board)
        else:
            self._has_finished = False
    
    @property
    def board_size(self):
        """Game board size"""
        return self._board_size
    
    @property
    def round(self):
        """Current round"""
        return self._round
    
    def board(self, view=BoardView.ONE_CHANNEL):
        """Create Othello board game representation

        Args:
            board_size ([BoardView]): Type of board visualization, 
        """
        if view == BoardView.TWO_CHANNELS:
            return self._board
        elif view == BoardView.ONE_CHANNEL:
            has_to_update = self._one_channel_board_last_update != self.round
            if has_to_update:
                self._one_channel_board = OthelloGame.convert_to_one_channel_board(self._board)
                self._one_channel_board_last_update = self.round 
            return self._one_channel_board

        raise TypeError('Expecting BoardView type')
    
    def is_square_free(self, x, y):
        """Check if there's not any piece on the board square

        Args:
            row ([int]): square row position
            col ([int]): square col position

        Returns:
            [bool]: True if the square is free, otherwise False
        """
        return OthelloGame.is_board_square_free(self._board, row, col)

    def is_valid_action(self, row, col):
        """Check the square is valid action for current player

        Args:
            row ([int]): square row position
            col ([int]): square col position

        Returns:
            [bool]: True if the square is a valid action, otherwise False
        """
        return OthelloGame.is_valid_player_action(self._board, self.current_player, row, col)
    
    def get_valid_actions(self):
        """Get all valid actions for current player

        Returns:
            [generator ndarray[(x, y), ...]]: All valid squares to current player play 
        """
        return OthelloGame.get_player_valid_actions(self._board, self.current_player)

    def get_free_squares(self):
        """Get all free squares on board

        Returns:
            [ndarray[(row, col), ...]]: All free square positions
        """
        return OthelloGame.get_board_free_squares(self._board)
    
    def has_finished(self):
        """Check if the game has finished

        Returns:
            [bool]: True if the games has finished, otherwise False
        """
        return self._has_finished
    
    def play(self, row, col):
        """Add new piece on board

        Args:
            row ([int]): Piece position on row
            col ([int]): Piece position on col
        """
        assert not self._has_finished, 'Game has ended'

        OthelloGame.flip_board_squares(self._board, self.current_player, row, col)
        
        self._round += 1
        
        self.current_player = self.current_player.opponent
    
        can_new_player_play = OthelloGame.has_player_actions_on_board(self._board, self.current_player)

        if not can_new_player_play:
            can_previous_player_play = OthelloGame.has_player_actions_on_board(self._board, self.current_player.opponent)

            if not can_previous_player_play:
                self._has_finished = True
            else:
                self.current_player = self.current_player.opponent
    
    def get_players_points(self):
        """Get players points

        Returns:
            [dict]: Points of each player
        """
        return OthelloGame.get_board_players_points(self._board)
    
    def get_winning_player(self):
        """Get winning player 

        Returns:
            [tuple]: ([OthelloPlayer], [int] Player points)
        """
        return OthelloGame.get_board_winning_player(self._board)

    @staticmethod
    def initial_board(board_size):
        assert board_size % 2 == 0, 'Board size must be even'
        
        initial = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=np.bool)
        pad = (board_size - 2) // 2

        return np.pad(initial, ((pad, pad), (pad, pad), (0, 0)), constant_values=0)
    
    @staticmethod
    def get_all_directions_squares(board_size, row, col):
        for direction in OthelloGame.ALL_DIRECTIONS:
            yield OthelloGame.get_direction_squares(board_size, direction, row, col) 

    @staticmethod
    def get_direction_squares(board_size, direction, row, col):
        row_offset, col_offset = direction
        row, col = row + row_offset, col + col_offset
        while 0 <= row < board_size and 0 <= col < board_size:
            yield row, col
            row += row_offset
            col += col_offset
    
    @staticmethod
    def get_board_free_squares(board):
        return np.argwhere(np.amax(board, axis=2) == 0)
    
    @staticmethod
    def is_board_square_free(board, row, col):
        return np.amax(board[row, col]) == 0
    
    @staticmethod
    def get_player_valid_actions(board, player):
        return (s for s in OthelloGame.get_board_free_squares(board) if OthelloGame.is_valid_player_action(board, player, *s))
    
    @staticmethod
    def is_valid_player_action(board, player, row, col):
        return next(OthelloGame.get_action_flip_squares(board, player, row, col), False) is not False
    
    @staticmethod
    def get_action_flip_squares(board, player, row, col):
        if not OthelloGame.is_board_square_free(board, row, col):
            return

        board_size = board.shape[0]

        opponent_channel = OthelloGame.PLAYER_CHANNELS[player.opponent]

        for direction_squares in OthelloGame.get_all_directions_squares(board_size, row, col):
            row_col = next(direction_squares, None)
            if row_col and board[row_col[0], row_col[1], opponent_channel]:
                flip_squares = [row_col]
                for row, col in direction_squares:
                    if OthelloGame.is_board_square_free(board, row, col):
                        break
                    elif board[row, col, OthelloGame.PLAYER_CHANNELS[player]]:
                        yield from flip_squares
                    else:
                        flip_squares.append((row, col))

    @staticmethod
    def flip_board_squares(board, player, row, col):
        player_channel = OthelloGame.PLAYER_CHANNELS[player]
        opponent_channel = OthelloGame.PLAYER_CHANNELS[player.opponent]

        for flip_row, flip_col in OthelloGame.get_action_flip_squares(board, player, row, col):
            board[flip_row, flip_col, player_channel] = 1
            board[flip_row, flip_col, opponent_channel] = 0
        
        board[row, col, player_channel] = 1
        board[row, col, opponent_channel] = 0
    
    @staticmethod
    def has_board_finished(board):
        can_black_play = OthelloGame.has_player_actions_on_board(board, OthelloPlayer.BLACK)
        return not can_black_play and not OthelloGame.has_player_actions_on_board(board, OthelloPlayer.WHITE)

    @staticmethod
    def get_board_winning_player(board):
        return max(OthelloGame.get_board_players_points(board).items(), key=lambda item: item[1])
    
    @staticmethod
    def get_board_players_points(board):
        return {p: np.count_nonzero(board[:, :, OthelloGame.PLAYER_CHANNELS[p]]) for p in OthelloPlayer}

    @staticmethod
    def has_player_actions_on_board(board, player):
        return next(OthelloGame.get_player_valid_actions(board, player), False) is not False

    @staticmethod
    def convert_to_one_channel_board(board):
        one_channel = board[:, :, OthelloGame.PLAYER_CHANNELS[OthelloPlayer.BLACK]] * OthelloPlayer.BLACK.value
        one_channel += board[:, :, OthelloGame.PLAYER_CHANNELS[OthelloPlayer.WHITE]] * OthelloPlayer.WHITE.value
        return one_channel

    @staticmethod
    def invert_board(board):
        return np.flip(board, axis=2)


def hash_ndarray(ndarray):
    """Generate hash to a ndarray

    Args:
        ndarray ([ndarray]): Numpy Array

    Returns:
        [int]: Hash of the numpy array
    """
    return hash(hashlib.sha1(np.ascontiguousarray(ndarray)).digest())


class MCTS:
    def __init__(self, degree_explorarion):
        self._Ns = {}
        self._Nsa = {}
        self._Qsa = {}
        self._Psa = {}

        self._state_actions = {}

        self.degree_explorarion = degree_explorarion
    
    def simulate(self, state):
        """Run an iteration of Monte Carlo Tree Search algorithm from a state

        Args:
            state ([ndarray]): State

        Returns:
            [float]: Simluated state value
        """
        if self.is_terminal_state(state):
            #print('final state')
            return -self.get_state_reward(state)

        hash_ = hash_ndarray(state)

        if hash_ not in self._Ns:
            #print("lance novo ---------------------------------------")
            self._Ns[hash_] = 0
            self._Nsa[hash_] = {}
            self._Qsa[hash_] = {} 
            self._Psa[hash_] = self.moves_scaled_by_valid_moves(state)
            sum_Psa = np.sum(self._Psa[hash_])
            if np.sum(self._Psa[hash_]) > 0:
                self._Psa[hash_] /= sum_Psa
            else:
                logging.warning("All valid moves were masked, doing a workaround.")
                self._Psa[hash_] = self._mask_valid_moves(state)
                self._Psa[hash_] /= np.sum(self._Psa[hash_])

            return -self.get_state_value(state)
        else:
            state_actions = self._get_state_actions(state)
            
            if not self._Nsa[hash_]:
                self._Nsa[hash_] = {a: 0 for a in state_actions}
                self._Qsa[hash_] = {a: 0 for a in state_actions}
            
            action = max(state_actions, key=lambda a: self._upper_confidence_bound(hash_, a))
            next_state = self.get_next_state(state, action)
            value = MCTS.simulate(self, next_state)
            #print(OthelloGame.convert_to_one_channel_board(state))
            #print(OthelloGame.convert_to_one_channel_board(next_state))
            self._Qsa[hash_][action] = (self._N(hash_, action) * self._Q(hash_, action) + value) / (self._N(hash_, action) + 1)
            self._Nsa[hash_][action] += 1
            self._Ns[hash_] += 1
            return -value
    
    def N(self, state, action=None):
        """Get number of visits during MCTS simulations

        Args:
            state ([ndarray]): State
            action ([Any], optional): Action. Defaults to None.

        Returns:
            [int]: Number of visits to the state (and action)
        """
        hash_ = hash_ndarray(state)
        return self._N(hash_, action)
    
    def is_terminal_state(self, state):
        """Check if the state is terminal

        Args:
            state ([ndarray]): State
        
        Returns:
            [bool]: True if the state is terminal, otherwise False.
        """
        raise NotImplementedError

    def get_state_value(self, state):
        """Get default state value

        Args:
            state ([ndarray]): State
        
        Returns:
            [float]: Default state value.
        """
        raise NotImplementedError

    def get_state_reward(self, state):
        """Get state reward

        Args:
            state ([ndarray]): Terminal state
        
        Returns:
            [float]: Terminal state reward.
        """
        raise NotImplementedError

    def get_state_actions_propabilities(self, state):
        """Get value of probabilities of all state's actions

        Args:
            state ([ndarray]): State
            action ([Any]): Action
        
        Returns:
            [dict]: Dictionary mapping actions to their probabilities
        """
        raise NotImplementedError
    
    def get_state_actions(self, state):
        """Get state's actions

        Args:
            state ([ndarray]): State
        
        Returns:
            [list]: State's actions
        """
        raise NotImplementedError
    
    def get_next_state(self, state, action):
        """Get next state from state and action

        Args:
            state ([ndarray]): State
            action ([Any]): Action
        
        Returns:
            [ndarray]: Next state from the state and the action
        """
        raise NotImplementedError

    def moves_scaled_by_valid_moves(self, state):
        """Get moves with probabilities scaled by valid moves

        Args:
            state ([ndarray]): State
        
        Returns:
            [matrix]: Matrix with action probabilities
        """
        raise NotImplementedError
    
    def _mask_valid_moves(self, state):
        raise NotImplementedError

    def _upper_confidence_bound(self, hash_, action):
        bound = math.sqrt(self._N(hash_)) / (1 + self._N(hash_, action))
        return self._Q(hash_, action) + self.degree_explorarion * self._P(hash_, action) * bound
    
    def _N(self, state_hash, action=None):
        if state_hash not in self._Ns: 
            return 0
        return self._Ns[state_hash] if action is None else self._Nsa[state_hash][action]
    
    def _P(self, state_hash, action):
        return self._Psa[state_hash][action]
    
    def _Q(self, state_hash, action):
        return self._Qsa[state_hash][action]

    def _get_state_actions(self, state):
        hash_ = hash_ndarray(state)
        if hash_ not in self._state_actions:
            self._state_actions[hash_] = self.get_state_actions(state)
        return self._state_actions[hash_]
