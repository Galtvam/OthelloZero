import math
import hashlib


class MCTS:
    def __init__(self, degree_explorarion):
        self._Ns = {}
        self._Nsa = {}
        self._Qsa = {}
        self._Psa = {}

        self.degree_explorarion = degree_explorarion
    
    def simulate(self, state):
        if self.is_terminal_state(state):
            return self.get_state_reward(state)

        hash_ = MCTS._hash_ndarray(state)

        if hash_ not in self._Ns:
            self._Ns[hash_] = 0
            self._Nsa[hash_] = {}
            self._Qsa[hash_] = {} 
            self._Ps[hash_] = self.get_state_actions_propabilities(state)
            return self.get_state_value(state)
        else:
            action = max(self.get_state_actions(state), key=lambda a: self._upper_confidence_bound(hash_, a))
            next_state = self.get_next_state(state, action)
            value = self.simulate(next_state)
            self._Qsa[hash_][action] = (self._N(hash_, action) * self._Q(hash_, action) + value) / (self._N(hash_, action) + 1)
            self._Nsa[hash_][action] += 1
            self._Ns[hash_] += 1
            return value
    
    def N(self, state, action=None):
        hash_ = MCTS._hash_ndarray(state)
        return self._N(state, action)
    
    def is_terminal_state(self, state):
        raise NotImplementedError

    def get_state_value(self, state):
        raise NotImplementedError

    def get_state_reward(self, state):
        raise NotImplementedError

    def get_state_actions_propabilities(self, state):
        raise NotImplementedError
    
    def get_state_actions(self, state):
        raise NotImplementedError
    
    def get_next_state(self, state, action):
        raise NotImplementedError

    def _upper_confidence_bound(self, hash_, action):
        bound = math.sqrt(self._N(hash_)) / (1 + self._N(hash_, action))
        return self._Q(hash_, action) + self.degree_explorarion * self._P(hash_, action) * bound
    
    def _N(self, hash_, action=None):
        return self._Ns[hash_] if action is None else self._Nsa[state][action]
    
    def _P(self, hash_, action):
        return self._Psa[hash_][action]
    
    def _Q(self, hash_, action):
        return self._Qsa[hash_][action]
    
    @staticmethod
    def _hash_ndarray(ndarray):
        return hash(hashlib.sha1(ndarray).digest())
