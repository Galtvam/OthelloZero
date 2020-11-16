# -*- coding: utf-8 -*-
"""
Imports
"""

from keras.models import Model
from keras.layers import Input, Reshape, Activation, Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam


"""
Model
"""

class OthelloNN:
  def __init__(self, board_size, num_channels_1, num_channels_2, lr=0.001, dropout=0.3):
    '''

    Inputs:
      board_size -> a tuple with the size of the board (n,n)
      lr -> a Scalar with the learning rate parameter
      dropout -> a Scalar with the dropout parameter
      num_channels_1 -> a Scalar with the number of filters in the first two ConvLayers
      num_channels_2 -> a Scalar with the number of filters in the last two ConvLayers

    Outputs:
      self.model -> a Keras model with the NN
      self.pi -> a Matrix of shape (batch_size x board_x x board_x) between [0,1]
      self.v -> a vector of Scalars with shape (batch_size x 1) between [-1,1]

    '''


    self.board_x, self.board_y = board_size
    self.action_size = self.board_x * self.board_y
    self.learning_rate = lr
    self.dropout = dropout
    self.num_channels_1 = num_channels_1
    self.num_channels_2 = num_channels_2


    self.input_boards = Input(shape=(self.board_x, self.board_y, 2)) #shape (batch_size, board_x, board_y, 2)

    conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels_1, 3, padding='same', use_bias=False)(self.input_boards))) #shape (batch_size, board_x, board_y, num_channels_1)
    conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels_1, 3, padding='same', use_bias=False)(conv1))) #shape (batch_size, board_x, board_y, num_channels_1)
    special1 = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(conv2) #shape (batch_size, board_x/2, board_y/2, num_channels_1)

    conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels_2, 3, padding='same', use_bias=False)(special1))) #shape (batch_size, board_x/2, board_y/2, num_channels_2)
    conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels_2, 3, padding='same', use_bias=False)(conv3))) #shape (batch_size, board_x/2, board_y/2, num_channels_2)

    flatten = Flatten()(conv4)  #shape (batch_size, board_x/2 x board_y/2 x num_channels_2)

    #value side
    v_dense1 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(flatten)))) # shape (batch_size x 512)
    v_dense2 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(256, use_bias=False)(v_dense1)))) # shpe (batch_size x 256)
    self.v = Dense(1, activation='tanh', name='v')(v_dense2) # shape (batch_size x 1)

    #pi side
    pi_dense1 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(flatten)))) # shape (batch_size x 512)
    pi_dense2 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(256, use_bias=False)(pi_dense1)))) # shape (batch_size x 256)
    self.pi = Dense(self.action_size, activation='softmax', name='pi')(pi_dense2) # shape (batch_size x action_size)
    self.pi = Reshape((self.board_x, self.board_y))(self.pi)

    self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
    self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(self.learning_rate))
