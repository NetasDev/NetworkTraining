from utils import *

import argparse
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from othello.OthelloGame import *
from tensorflow.keras import backend as K
import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
import tensorflow as tf
from wandb.keras import WandbCallback
import wandb
from utils import *
from NeuralNet import NeuralNet

import argparse





class OthelloNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        """(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)"""
        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = OthelloNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        history = self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)
        return history.history['loss'][-1],history.history['pi_loss'][-1],history.history['v_loss'][-1]
        

        
    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()
        #np.set_printoptions(threshold=np.inf)

        # preparing input
        board = board[np.newaxis, :, :]

        # run   
        pi, v = self.nnet.model.predict(board)

        outputs = [layer.output for layer in self.nnet.model.layers][1:]        # all layer outputs except first (input) layer
        functor = K.function([self.nnet.model.input, K.learning_phase()], outputs )
        layer_outs = functor([board, 1.])
        print(layer_outs[1])
        for layer in self.nnet.model.layers:
            print(layer.name)
        for layer_out in layer_outs:
            print(layer_out.shape)
        #   outputs = [K.function([self.nnet.model.input], [layer.output])([board])[1:] for layer in self.nnet.model.layers]


        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint'):
        filepath = os.path.join(folder, filename)

        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
 
        self.nnet.model.save(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.nnet.model = tf.keras.models.load_model(filepath)


nnet = NNetWrapper(OthelloGame(6))
nnet.load_checkpoint(folder='./temp/EPStest/40/',filename="best")
ara =np.array([[ 0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0],
                [ 0,  0, -1,  1,  0,  0],
                [ 0,  0,  1, -1,  0,  0],
                [ 0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0]])
pi = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23529411764705882, 0.0, 0.0, 0.0, 0.0, 0.2647058823529412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2647058823529412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23529411764705882, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
example = (ara,pi,1)
nnet.predict(ara)

