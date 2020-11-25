import argparse

from Othello import OthelloGame, OthelloPlayer, BoardView
from Net.NNet import NNetWrapper
from training import OthelloMCTS

DEFAULT_CHECKPOINT_FILEPATH = './othelo_model.weights'


def valid_move(possible_moves, row, column):
    '''
    valids the move makes for the player
    '''
    possible_moves = (tuple(possible_moves))
    move = (row, column)
    flag = False
    for m in possible_moves:
        print(tuple(m), tuple(move))
        if tuple(m) == tuple(move):
            flag = True
            print(flag)
            return flag
    print("JOGA A JOGADA DIREITO SEU MERDINHA")
    return flag

def move_in_terminal(game):
    '''
    Takes the user's input in the terminal and 
    returns the line and column of the move
    '''
    
    is_valid = False
    while not is_valid:
        possible_moves = game.get_valid_actions()
        print(game.board(view=BoardView.ONE_CHANNEL))
        your_move = input(str("digite a linha e a coluna respectivamente nesta ordem\nSua jogada>>> "))
        row,column = int(your_move[0])-1, int(your_move[1]) -1 if your_move[1] != " " else int(your_move[2]) -1
        is_valid = valid_move(possible_moves,row, column)
    return row, column

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

def playing_with_machine(board_size, neural_network, degree_exploration=1, num_simulations=25, human_player=1 ):

    human_player = OthelloPlayer.BLACK if human_player == 1 else OthelloPlayer.WHITE
    machine_player = OthelloPlayer.WHITE if human_player is OthelloPlayer.BLACK else OthelloPlayer.BLACK
    
    game = OthelloGame(board_size, current_player = OthelloPlayer.BLACK)
    neural_networks_mcts = OthelloMCTS(board_size, neural_network, degree_exploration)
    while not game.has_finished():

        #human move
        if game.current_player is human_player:
            row, column = move_in_terminal(game)
            game.play(row, column)

        #machine move
        else:
            machine_move(game, neural_networks_mcts, num_simulations)
            
    winner, points = game.get_winning_player()       
    print (f'O jogador {winner} ganhou com {points} pontos')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', '--weights-file', default=DEFAULT_CHECKPOINT_FILEPATH, help='File path to load neural network weights')
    parser.add_argument('-b', '--board-size', default=6, type=int, help='Othello board size')
    parser.add_argument('-c', '--constant-upper-confidence', default=1, type=int, help='MCTS upper confidence bound constant')
    parser.add_argument('-s', '--simulations', default=25, type=int, help='Number of MCTS simulations by episode')
    parser.add_argument('-p', '--player', default=1, type=int, help='Color of player pieces, number 1 represents the black pieces and -1 represents the white pieces')
    args = parser.parse_args()
    neural_network = NNetWrapper(board_size=(6,6))
    if args.weights_file:
        neural_network.load_checkpoint(args.weights_file)

    playing_with_machine(args.board_size, neural_network, args.constant_upper_confidence, args.simulations, args.player)
