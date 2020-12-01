# -*- coding: utf-8 -*-
"""
Imports
"""
import numpy as np
import os
import datetime
import tensorflow as tf

from enum import Enum, auto

from .OthelloNN import OthelloNN as onn
from .BaseNN import BaseNN as bnn

class NeuralNets(Enum):
    ONN = auto()
    BNN = auto()


"""
Wrapper
"""
class NNetWrapper:
    def __init__(self, board_size=(8,8), batch_size=32, epochs=10,
                 num_channels_1=128, num_channels_2=256, lr=0.001, dropout=0.3, network=NeuralNets.ONN):
        '''
        Inputs:
          board_size -> a Tuple with the size of the board (n,n)
          batch_size -> a Scalar with the batch size used for training
          epochs -> a Scalar with the number of epochs during training
          num_channels_1 -> a Scalar with the number of channels for the first 2 layers in ONN
          num_channels_2 -> a Scalar with the number of channels for the last 2 layers in ONN
          lr -> a Scalar with the learning rate to use during training
          dropout -> a Scalar with the dropout rate of the network
          network -> a NeuralNets item with the respective tipe of network
                      > ONN: New architecture, more robust
                      > BNN: Base architecture from the original paper 

        '''
        self.board_size_x, self.board_size_y = board_size
        self.action_size = self.board_size_x * self.board_size_y
        self.batch_size = batch_size
        self.epochs = epochs
        self.network_type = network

        if self.network_type == NeuralNets.ONN:
          self.nnet = onn(board_size=board_size, num_channels_1=num_channels_1, 
                          num_channels_2=num_channels_2, lr=lr, dropout=dropout)
        elif self.network_type ==  NeuralNets.BNN:
          self.nnet = bnn(board_size=board_size, lr=lr, dropout=dropout)
        else:
          return Exception('Invalid Network Type.')

    def train(self, examples, verbose=None):
        '''
        Inputs:
          examples -> a List with examples used for training, shape = (number_examples, board_size_x, board_size_y, 2)

        '''
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        return self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=self.batch_size, 
                                   epochs=self.epochs, verbose=verbose, callbacks=[tensorboard_callback])

    def predict(self, board):
        '''
        Inputs:
          board -> a Matrix representing the board state, shape= (board_size_x, board_size_y, 2)

        Output:
          pi -> a Vector representing the policy with the propabilities for each position, shape= (action_size)
          v -> a Scalar with the value predicted for the given board, range= [-1, 1]

        '''
        if self.network_type == NeuralNets.ONN:
          board = board[np.newaxis, :, :, :]
        else:
          board = board[np.newaxis, :, :]

        pi, v = self.nnet.model.predict(board)
        pi = pi.reshape(self.board_size_x, self.board_size_y)
        return pi, v[0][0]

    # save weights
    def save_checkpoint(self, filepath):
        self.nnet.model.save_weights(filepath, save_format='h5')

    # load saved weights
    def load_checkpoint(self, filepath):
        assert filepath.endswith('.h5'), 'Expecting a file with .h5 as extension'
        self.nnet.model.load_weights(filepath)

    def copy(self):
        copy_wrapper = NNetWrapper((self.board_size_x, self.board_size_y), network=self.network_type)
        copy_wrapper.nnet.model.set_weights(self.nnet.model.get_weights())
        return copy_wrapper
      
