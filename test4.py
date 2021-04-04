
from othello.OthelloGame import *
import numpy as np
from time import perf_counter

game = OthelloGame(8)
start_board = np.array(([ 1, 1,-1, 0, 0, 0, 0, 1],
                        [ 0, 0, 0, 0, 0, 0, 0, 1],
                        [ 0, 0, 0, 0, 1, 1, 0, 1],
                        [ 0, 0, 0,-1, 1, 1, 0,-1],
                        [ 0, 0, 0,-1,-1, 1, 0,-1],
                        [ 0, 0, 0, 0, 0, 0, 0,-1],
                        [ 0, 0, 0, 0, 0, 0, 0,-1],
                        [ 0, 0, 0,-1, 0, 0, 0,-1]))

time = perf_counter()
print(game.get_better_value(start_board,1))
print(perf_counter()-time)