import math
import hashlib
import numpy as np
import logging


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
            return -self.get_state_reward(state)

        hash_ = hash_ndarray(state)

        if hash_ not in self._Ns:
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
