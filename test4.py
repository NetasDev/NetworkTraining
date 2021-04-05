
from othello.OthelloGame import *

from othello.OthelloPlayers import *
import numpy as np
from time import perf_counter

game = OthelloGame(8)
start_board = np.array(([ 0, 0, 1, 0, 0,-1,-1, 0],
                        [-1, 0, 1, 1, 1,-1, 0, 0],
                        [-1,-1,-1,-1,-1, 1, 0, 0],
                        [-1,-1, 1, 1, 1, 1, 1, 0],
                        [-1,-1,-1,-1,-1,-1, 1, 0],
                        [-1,-1,-1,-1,-1,-1,-1, 0],
                        [-1,-1,-1,-1, 1,-1,-1, 0],
                        [ 0, 0,-1,-1,-1,-1,-1, 0]))

time = perf_counter()
#print(game.get_better_value(start_board,1))
#print(perf_counter()-time)
start_board =game.getCanonicalForm(start_board,-1)

minimax = MinimaxOthelloPlayer(game,5,mode=2,maxtime=2)
minimax2 = MinimaxOthelloPlayer(game,5,mode=3,maxtime=2)

print(minimax.play(start_board,details=True))
print(minimax2.play(start_board,details=True))