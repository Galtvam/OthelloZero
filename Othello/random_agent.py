import random

from Othello import OthelloGame, OthelloPlayer, BoardView

def random_agent(game):

    '''
    Makes random moves
    '''

    possible_moves = game.get_valid_actions()
    move = random.choice(tuple(possible_moves))
    game.play(move[0],move[1])
