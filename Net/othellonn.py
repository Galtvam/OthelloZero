# -*- coding: utf-8 -*-
"""
Imports
"""

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Reshape, Activation, Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam

"""Model:"""

class OthelloNN:
  def __init__(self, game, args):
    '''

    Inputs:
      game -> representative class of the game configurations
      args -> dict with num_channels_1, num_channels_2, dropout, lr (learning rate)

    Outputs:
      self.model -> a Keras model with the NN
      self.pi -> a vector of shape (batch_size x action_size) between [0,1]
      self.v -> a vector of Scalars with shape (batch_size x 1) between [-1,1]

    '''


    self.board_x, self.board_y = game.getBoardSize()
    self.action_size = game.getActionSize()
    self.args = args

    self.input_boards = Input(shape=(self.board_shape)) #shape (board_x, board_y, 1)

    x_input = Reshape((self.board_x, self.board_y, 1))(self.input_boards) #shape (batch_size, board_x, board_y, 1)


    conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels_1, 3, padding='same', use_bias=False)(x_image))) #shape (batch_size, board_x, board_y, num_channels_1)
    conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels_1, 3, padding='same', use_bias=False)(conv1))) #shape (batch_size, board_x, board_y, num_channels_1)
    special1 = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(conv2) #shape (batch_size, board_x/2, board_y/2, num_channels_1)

    conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels_2, 3, padding='same', use_bias=False)(special1))) #shape (batch_size, board_x/2, board_y/2, num_channels_2)
    conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels_2, 3, padding='same', use_bias=False)(conv3))) #shape (batch_size, board_x/2, board_y/2, num_channels_2)

    flatten = Flatten()(conv4)  #shape (batch_size, board_x/2 x board_y/2 x num_channels_2)

    #value side
    v_dense1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(flatten)))) # shape (batch_size x 512)
    v_dense2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(256, use_bias=False)(v_dense1)))) # shpe (batch_size x 256)
    self.v = Dense(1, activation='tanh', name='v')(v_dense2) # shape (batch_size x 1)

    #pi side
    pi_dense1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(flatten)))) # shape (batch_size x 512)
    pi_dense2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(256, use_bias=False)(pi_dense1)))) # shape (batch_size x 256)
    self.pi = Dense(self.action_size, activation='softmax', name='pi')(pi_dense2) # shape (batch_size x action_size)

    self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
    self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
