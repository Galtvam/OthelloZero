import random
import logging
import numpy as np

from Net.NNet import NeuralNets
from Othello import OthelloGame, OthelloPlayer, BoardView

from othelo_mcts import OthelloMCTS 


class OthelloAgent:
    def __init__(self, game):
        self.game = game
    
    def play(self):
        """Do an action on OthelloGame"""
        raise NotImplementedError


class RandomOthelloAgent(OthelloAgent):
    def play(self):
        possible_moves = tuple(self.game.get_valid_actions())
        move = random.choice(possible_moves)
        self.game.play(*move)


class GreedyOthelloAgent(OthelloAgent):
    def play(self):
        move_points = {}
        possible_moves = tuple(self.game.get_valid_actions())
        points_before = game.get_players_points()[game.current_player]
        board = self.game.board(BoardView)
        
        for move in possible_moves:
            state = np.copy(self.game.board(BoardView))
            OthelloGame.flip_board_squares(state, game.current_playe, *move)
            points = OthelloGame.get_board_players_points(state)[OthelloPlayer.BLACK] - points_before
            move_points[move] = points
        
        greedy_move = max(move_points, key=move_points.get)
        game.play(*greedy_move)


class NeuralNetworkOthelloAgent(OthelloAgent):
    def __init__(self, game, neural_network, num_simulations, degree_exploration, temperature=0):
        self.temperature = 0
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.mcts = OthelloMCTS(game.board_size, neural_network, degree_exploration)
        super().__init__(game)
    
    def play(self):
        state = self.game.board(BoardView.TWO_CHANNELS)
        for _ in range(self.num_simulations):
            self.mcts.simulate(state, self.game.current_player)

        if self.game.current_player == OthelloPlayer.WHITE:
            state = OthelloGame.invert_board(state)
        
        if self.neural_network.network_type is NeuralNets.ONN:
            action_probabilities = self.mcts.get_policy_action_probabilities(state, self.temperature)
        else:
            action_probabilities = self.mcts.get_policy_action_probabilities(
                self.game.board(), self.temperature)

        valid_actions = self.game.get_valid_actions()
        best_action = max(valid_actions, key=lambda position: action_probabilities[tuple(position)])
        self.game.play(*best_action)


def duel_between_agents(game, agent_1, agent_2):
    players_agents = {
        OthelloPlayer.BLACK: agent_1,
        OthelloPlayer.WHITE: agent_2
    }

    logging.info(f'Duel - Started')
    while not game.has_finished():
        logging.info(f'Duel - Round: {game.round}')
        agent = players_agents[game.current_player]
        agent.play()

    winner, points = game.get_winning_player()
    return players_agents[winner], points
