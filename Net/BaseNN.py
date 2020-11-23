# -*- coding: utf-8 -*-
"""
Imports
"""

from keras.models import Model
from keras.layers import Input, Reshape, Activation, Dense, Conv2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam

"""
Model
"""

class BaseNN:
    def __init__(self, board_size, num_channels=512, lr=0.001, dropout=0.3):
        '''

        Inputs:
            board_size -> a tuple with the size of the board (n,n)
            lr -> a Scalar with the learning rate parameter
            dropout -> a Scalar with the dropout parameter
            num_channels -> a Scalar with the number of filters in the ConvLayers

        Outputs:
            self.model -> a Keras model with the NN
            self.pi -> a Matrix of shape (batch_size x board_x x board_x) between [0,1]
            self.v -> a vector of Scalars with shape (batch_size x 1) between [-1,1]

        '''

        # game params

        self.board_x, self.board_y = board_size
        self.action_size = self.board_x * self.board_y
        self.learning_rate = lr
        self.dropout = dropout
        self.num_channels = num_channels


        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same')(x_image)))         
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same')(h_conv1)))         
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='valid')(h_conv2)))        
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='valid')(h_conv3)))       
        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat)))) 
        s_fc2 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))          
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)
        self.pi = Reshape((self.board_x, self.board_y))(self.pi)  
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(self.learning_rate))