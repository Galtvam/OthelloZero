import numpy as np

from OthelloNN import OthelloNN as onn


class NNetWrapper():
    def __init__(self, board_size=(8,8), batch_size=32, epochs=10):
        self.board_size_x, self.board_size_y = board_size
        self.nnet = onn(board_size=board_size, num_channels_1=64, num_channels_2=128, lr=0.001, dropout=0.3)
        self.action_size = self.board_size_x * self.board_size_y
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, examples):
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = self.batch_size, epochs = self.epochs)

    def predict(self, board):
        board = board[np.newaxis, :, :, :]
        pi, v = self.nnet.model.predict(board)
        return pi[0], v[0]
