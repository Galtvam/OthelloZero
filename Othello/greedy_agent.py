import numpy as np

from Othello import OthelloGame, OthelloPlayer, BoardView

def points_after_move (state, move):
    '''
    creates a copy of the game instance and returns ]
    the amount of points after the move
    '''

    board = np.copy(state)
    print(move)
    OthelloGame.flip_board_squares(board, OthelloPlayer.BLACK, move[0], move[1])
    points_after = OthelloGame.get_board_players_points(board)[OthelloPlayer.BLACK]
    if OthelloGame.has_player_actions_on_board(board, OthelloPlayer.WHITE):
        # Invert board to keep using BLACK perspective
        board = OthelloGame.invert_board(board)
   
    return points_after
    

def greedy_agent(game):

    '''
    Makes the move that takes more opponent's pieces in the round
    '''

    move_points = {} 
    possible_moves = game.get_valid_actions()
    points_before = game.get_players_points()[game.current_player]
    state = game.board(BoardView.TWO_CHANNELS)
    for move in possible_moves:
        move = tuple(move)
        points_after = points_after_move(state, move)
        points = points_after - points_before
        move_points[move] = points

    print(move_points)
    greedy_move = max(move_points, key=move_points.get)
    print("Jogada escolhida foi",greedy_move)
    game.play(greedy_move[0], greedy_move[1])
