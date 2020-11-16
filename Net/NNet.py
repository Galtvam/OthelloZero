# -*- coding: utf-8 -*-
"""
Imports
"""
import numpy as np
import os

from .OthelloNN import OthelloNN as onn


"""
Wrapper
"""
class NNetWrapper:
    def __init__(self, board_size=(8,8), batch_size=32, epochs=10,
                 num_channels_1=64, num_channels_2=128, lr=0.001, dropout=0.3):
        '''
        Inputs:
          board_size -> a Tuple with the size of the board (n,n)
          batch_size -> a Scalar with the batch size used for training
          epochs -> a Scalar with the number of epochs during training

        '''
        self.board_size_x, self.board_size_y = board_size
        self.action_size = self.board_size_x * self.board_size_y
        self.batch_size = batch_size
        self.epochs = epochs

        self.nnet = onn(board_size=board_size, num_channels_1=num_channels_1, 
                        num_channels_2=num_channels_2, lr=lr, dropout=dropout)

    def train(self, examples, verbose=None):
        '''
        Inputs:
          examples -> a List with examples used for training, shape = (number_examples, board_size_x, board_size_y, 2)

        '''
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        
        return self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=self.batch_size, 
                                   epochs=self.epochs, verbose=verbose)

    def predict(self, board):
        '''
        Inputs:
          board -> a Matrix representing the board state, shape= (board_size_x, board_size_y, 2)

        Output:
          pi -> a Vector representing the policy with the propabilities for each position, shape= (action_size)
          v -> a Scalar with the value predicted for the given board, range= [-1, 1]

        '''
        board = board[np.newaxis, :, :, :]
        pi, v = self.nnet.model.predict(board)
        pi = pi.reshape(self.board_size_x, self.board_size_y)
        return pi, v[0][0]

    # save weights
    def save_checkpoint(self, filepath):
        self.nnet.model.save_weights(filepath)

    # load saved weights
    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
    
    def copy(self):
        copy_wrapper = NNetWrapper((self.board_size_x, self.board_size_y))
        copy_wrapper.nnet.model.set_weights(self.nnet.model.get_weights())
        return copy_wrapper
