import Arena
import numpy as np
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.keras.NNet import NNetWrapper as nn
from othello.OthelloInteractiveBoard import InteractiveBoard
from othello.OthelloLogic import Board
from othello.OthelloPlayers import *
import pygame

import numpy as np
from utils import *

"""
TK This script is set up to easily use the functions provided by the Project
"""

"""
TK First an object of the game has to be created and atleast 2 Players of that game in order to play the game.
"""
game = OthelloGame(6)
hp = HumanOthelloPlayer(game)
greed = GreedyOthelloPlayer(game)
minimax = MinimaxOthelloPlayer(game,4)
minimax.name = "valueMatrix"
minimax2 = MinimaxOthelloPlayer(game,4,mode=1)
minimax2.name = "baseMinimax"
minimax3 = MinimaxOthelloPlayer(game,4,mode=2)
minimax3.name = "betterFunction"
"""
TK In Order to create a neural Network player with MCTS a NNetWrapper has to be created and this Wrapper has to load the net 
from the target folder with the given filename
Also a dotdict including atleast numMCTSSims (the number of MCTS Simulations done each turn) and cpuct(a factor for exploration, normally 1)
has to be created.
"""
"""
network = nn(game)
network.load_checkpoint(folder="./temp/Othello8x8/EPStest/",filename="best")
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
neuralplayer = NeuralNetworkPlayer(game,network,args)

network2 = nn(game)
network2.load_checkpoint(folder="./temp/new",filename="checkpoint_12")
neuralplayer2 = NeuralNetworkPlayer(game,network2,args)
"""
"""
Tk Once the Players are created, they can be matched against each other 1vs1 by creating an arena Object
This Object needs both players and the game to be played to be initialized.
Additionally a number det_turns can be set to make the players explore for the first det_turns instead of always choosing
the best found solution
"""
#arena = Arena.Arena(minimax,minimax2,game,det_turns=3)
"""
TK Afterwards the function playGames can be used to play X games between the two players.
X has to be a multiple of 2
The variable Interactive can be set to True to have an interactive board shown on screen.
This Board shows the current board, the possible moves of the current player und more information about the match.
"""
#player1wins,player2wins,draws = arena.playGames(10,Interactive=True,save="./newFolder/testgames2/")
#print(arena.player1.name + " : "+ str(player1wins))

#print(arena.player2.name + " : "+ str(player2wins))
#print("draws : " +str(draws))
"""
"""
TK By setting save to a path all games played will by saved in the folder at the given path.   
Afterwards they can be loaded and shown as a replay
"""

InBoard = InteractiveBoard.load("./newFolder/testgames2/game0")
InBoard.show_replay()

"""
TK There is also the Option to play a tournament between two or more players
If a save path is set for the tournament,there will be new folders created with the names of the two players
matched against each other and the games will be saved in there.
"""
"""
network = nn(game)
network.load_checkpoint(folder="./temp/Othello8x8/EPStest/40/",filename="best")
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
neuralplayer = NeuralNetworkPlayer(game,network,args)
neuralplayer.name = "40EPS"

network2 = nn(game)
network2.load_checkpoint(folder="./temp/Othello8x8/EPStest/80/",filename="best")
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
neuralplayer2 = NeuralNetworkPlayer(game,network2,args)
neuralplayer2.name = "80EPS"

network3 = nn(game)
network3.load_checkpoint(folder="./temp/Othello8x8/EPStest/160/",filename="best")
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
neuralplayer3 = NeuralNetworkPlayer(game,network3,args)
neuralplayer3.name = "160EPS"

arena = Arena.Arena(neuralplayer,neuralplayer3,game,3)
network1wins,network2wins,draws =arena.playGames(100,save="newFolder/soontobedeleted/")
print(network1wins)
print(network2wins)
print(draws)
"""

"""
players = []
players.append(neuralplayer)
players.append(neuralplayer2)
players.append(neuralplayer3)
Arena.Arena.play_tournament(players,40,game,2,savefolder="./newFolder/tournament")
"""


"""
folder = "./temp/Othello6x6/FirstModel/"
network = nn(game)
network.load_checkpoint(folder=folder,filename="best")
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
neuralplayer = NeuralNetworkPlayer(game,network,args)
neuralplayer.name = "16h_player"


Arena.Arena.play_one_against_many(neuralplayer,folder,200,game,8,savefolder="./one_many_2/")
"""

"""
network = nn(game)
folder = "./temp/Othello6x6/SecondModel/"
network.load_checkpoint(folder=folder,filename="best")
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
bestplayer = NeuralNetworkPlayer(game,network,args,name="best player")

folder = "./temp/Othello6x6/TryoutModel/"
network2=nn(game)
network2.load_checkpoint(folder=folder,filename="best")
worstplayer = NeuralNetworkPlayer(game,network2,args,name="worst player")

arena = Arena.Arena(bestplayer,worstplayer,game,tempThreshold=8)
print(arena.playGames(600,save = "./newTry/"))
"""

"""
Inboard = InteractiveBoard.load("./newTry/game6")
Inboard.show_replay()
"""

network = nn(game)
folder = "./temp/Othello6x6/LearningRate/001/"
network.load_checkpoint(folder=folder,filename="best")
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
bestplayer = NeuralNetworkPlayer(game,network,args,name="best player")