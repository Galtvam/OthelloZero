from Othello import OthelloGame, OthelloPlayer

def greedy_agent(game):
    
    '''
    Makes greedy moves
    '''

    greedy_color = game.current_player
    possible_moves = game.get_valid_actions()
    moves_result = []
    
    for move in possible_moves:
        temp_game = game.copy()
        temp_game.play(move[0], move[1])
        points = temp_game.get_players_points()
        moves_result.append(tuple(points[greedy_color], move))
    
    moves_result.sort()
    greedy_move = moves_result[0][1]
    game.play(greedy_move[0], greedy_move[1])
