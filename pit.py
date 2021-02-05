import Arena
import numpy as np
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.keras.NNet import NNetWrapper as nn

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
g = OthelloGame(6)

hp = HumanOthelloPlayer(g)
greed = GreedyOthelloPlayer(g)
minimax = MinimaxPlayer(g,4)

network = nn(g)
network.load_checkpoint(folder="./temp/new",filename="best")
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
neuralplayer = NeuralNetworkPlayer(g,network,args)

network2 = nn(g)
network2.load_checkpoint(folder="./temp/new",filename="checkpoint_12")
neuralplayer2 = NeuralNetworkPlayer(g,network2,args)
"""
arena = Arena.Arena(minimax,greed,g,det_turns=3,display=OthelloGame.display)
player1wins,player2wins,draws = arena.playGames(20,verbose=True)
print(player1wins)
print(player2wins)
print(draws)
"""
players=[]
players.append(minimax)
players.append(greed)
players.append(neuralplayer)
players.append(neuralplayer2)



#print(arena.playGames(6, verbose=True))

def play_tournament(players,num_matches,game,det_turns):
    wins = np.zeros((len(players),len(players)),dtype=int)
    draws = np.zeros((len(players),len(players)),dtype=int)

    for i in range(len(players)):
        for j in range(len(players)):
            if i!=j:
                arena = Arena.Arena(players[i],players[j],game,det_turns=det_turns)
                wins[i][j],_,draws[i][j] = arena.playGames(num_matches)
        
    np.savetxt('wins',wins,delimiter = " ")
    np.savetxt('draws',draws,delimiter=" ")

play_tournament(players,4,g,2)

            

