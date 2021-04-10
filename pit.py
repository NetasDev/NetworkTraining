import Arena
import numpy as np
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.keras.NNet import NNetWrapper as nn
from othello.OthelloInteractiveBoard import InteractiveBoard
from othello.OthelloLogic import Board
from othello.OthelloPlayers import *
import pygame
import os
import wandb

import numpy as np
from utils import *

"""
TK This script is set up to easily use the functions provided by the Project
"""

"""
TK First an object of the game has to be created and atleast 2 Players of that game in order to play the game.
"""
"""

hp = HumanOthelloPlayer(game)
greed = GreedyOthelloPlayer(game)
minimax = MinimaxOthelloPlayer(game,4)
minimax.name = "valueMatrix"
minimax2 = MinimaxOthelloPlayer(game,4,mode=1)
minimax2.name = "baseMinimax"
minimax3 = MinimaxOthelloPlayer(game,4,mode=2)
minimax3.name = "betterFunction"
"""

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
TK By setting save to a path all games played will by saved in the folder at the given path.   
Afterwards they can be loaded and shown as a replay
"""
"""
InBoard = InteractiveBoard.load("./newFolder/testgames2/game0")
InBoard.show_replay()
"""
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



"""
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
network01 = nn(game)
network01.load_checkpoint(folder="./temp/Othello8x8/LearningRate/001/",filename="best")
neuralplayer1 = NeuralNetworkPlayer(game,network01,args,name="0.1",maxtime=1)

minimax = MinimaxOthelloPlayer(game,20,maxtime=1)
minimax.name = "valueMatrix"

arena = Arena.Arena(neuralplayer1,minimax,game,tempThreshold=5)
networkswins, minimaxwins,_ =arena.playGames(10,save="./quicktest4/")
print(networkswins)
print(minimaxwins)
"""
#########
"""

game = OthelloGame(8)
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
minimax = MinimaxOthelloPlayer(game,3)

network01 = nn(game)
network01.load_checkpoint(folder='./temp/Othello8x8/MCTS sims/25/',filename="best")
neuralplayer1 = NeuralNetworkPlayer(game,network01,args,name="MCTS sims 25")

network001 = nn(game)
network001.load_checkpoint(folder='./temp/Othello8x8/MCTS sims/50/',filename="best")
neuralplayer2 = NeuralNetworkPlayer(game,network001,args,name="MCTS sims 50")


network0001 = nn(game)
network0001.load_checkpoint(folder='./temp/Othello8x8/MCTS sims/100/',filename="best")
neuralplayer3 = NeuralNetworkPlayer(game,network0001,args,name="MCTS sims 100")

players = []
players.append(neuralplayer1)
players.append(neuralplayer2)
players.append(neuralplayer3)

Arena.Arena.play_tournament(players,200,game,15,savefolder="./tournament3/Othello8x8/MCTS sims")
"""
"""
InBoard = InteractiveBoard.load("./testgames4/game47")
InBoard.show_replay()
"""
"""
InBoard = InteractiveBoard.load("./tournament/Othello8x8/Learning rate/lr 0.1 VS lr 0.01/game2")
print(InBoard.prediction_history)

sum1 = 0
sum2 = 0
turn = 0
for entry in InBoard.prediction_history:
    if turn%2 == 1:
        sum1 += entry[2]
    else:
        sum2 += entry[2]
    turn += 1
print(sum1*2/turn)
print(sum2*2/turn)
print(InBoard.action_history)
InBoard.show_replay()
"""

"""
args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
game = OthelloGame(8)
network = nn(game)
network.load_checkpoint(folder='./temp/Othello8x8/Continued model/',filename="best")
neuralplayer = NeuralNetworkPlayer(game,network,args,name="final model")
Arena.Arena.play_one_against_many(neuralplayer,"./temp/Othello8x8/Continued model/",100,game,15,savefolder="./previous generations 8x8 continued/")
"""
#####generate data
"""
args = dotdict({'numMCTSSims': 5000000, 'cpuct': 1.0})
game = OthelloGame(6)
network = nn(game)
network.load_checkpoint(folder='./temp/Othello6x6/Continued model/',filename="best")
neuralplayer = NeuralNetworkPlayer(game,network,args,maxtime=1)
randomplayer = RandomPlayer(game)
arena = Arena.Arena(neuralplayer,randomplayer,game,tempThreshold=8)
wins,losses,draws =arena.playGames(20)

print("wins: "+str(wins))
print("losses: "+str(losses))
print("draws: "+str(draws))
"""

"""
args = dotdict({'numMCTSSims': 5000000, 'cpuct': 1.0})
game = OthelloGame(8)
network = nn(game)
network.load_checkpoint(folder='./temp/Othello8x8/Continued model/',filename="best")
neuralplayer = NeuralNetworkPlayer(game,network,args,maxtime=1)
neuralplayer.name = "neural network"
minimax = MinimaxOthelloPlayer(game,1000,mode=1,maxtime=1)
minimax.name ="minimax value matrix"
arena = Arena.Arena(neuralplayer,minimax,game,tempThreshold=8)
wins,losses,draws =arena.playGames(400,save="./evaluation_games/8x8vsminimax/coin value with deter/")

print("wins: "+str(wins))
print("losses: "+str(losses))
print("draws: "+str(draws))
"""
"""
Inboard = InteractiveBoard.load("./evaluation_games/8x8vsminimax/combined value/game1")
Inboard.show_replay()

Inboard = InteractiveBoard.load("./evaluation_games/8x8vsminimax/combined value/game2")
Inboard.show_replay()

Inboard = InteractiveBoard.load("./evaluation_games/8x8vsminimax/combined value/game3")
Inboard.show_replay()
"""
######## look at games
"""
Inboard = InteractiveBoard.load("./evaluation_games/")
Inboard.show_replay()
Inboard = InteractiveBoard.load("./first runsgame2")
Inboard.show_replay()
Inboard = InteractiveBoard.load("./first runsgame3")
Inboard.show_replay()
"""
"""
game = OthelloGame(6)
minimax = MinimaxOthelloPlayer(game,10,mode=2,maxtime=1)
minimax.name = "better evaluation"
minimax2 = MinimaxOthelloPlayer(game,10,mode=3,maxtime=1)
arena = Arena.Arena(minimax,minimax2,game,tempThreshold=8)
wins,losses,draws =arena.playGames(20,save="./temporary3/")
print(str(wins)+" "+str(losses)+" "+str(draws))
"""
"""
game = OthelloGame(6)
hp = HumanOthelloPlayer(game)
arena = Arena.Arena(hp,hp,game)
arena.playGames(10,Interactive=True)
"""
"""
i = 1
while(True):
    
    Inboard = InteractiveBoard.load("./evaluation_games/8x8vsminimax/value matrix with deter/game"+str(i))
    Inboard.show_replay()
    i = i+1
"""

########### data analysis


all_mcts_simulations = 0
evaluation_mcts = np.zeros((140,3))
evaluation_minimax = np.zeros((140,3))

average_board_value = np.zeros((140,4))

score_difference = np.zeros((140,2))
wins,losses,draws = 0,0,0

heatmap_neural = np.zeros((8,8))

for i in range(1000):
    path = "./evaluation_games/8x8vsminimax/value matrix with deter/game"+str(i)
    path2 = "./evaluation_games/8x8vsminimax/coin value with deter/game"+str(i)
    path3 =  "./evaluation_games/8x8vsminimax/combined value with deter/game"+str(i)

    if os.path.isfile(path+".pkl"):
        Inboard = InteractiveBoard.load(path)
        Inboard2 = InteractiveBoard.load(path2)
        Inboard3 = InteractiveBoard.load(path3)
        neural = 0

        if Inboard.player1_name == "neural network":
            neural = 1
        else:
            neural = -1
        for j in range(len(Inboard.board_history)):
            score_difference[j][0] += Inboard.game.getScore(Inboard.board_history[j],neural)
            score_difference[j][1] += 1
            average_board_value[j][0]+= Inboard.game.get_coin_value(Inboard.board_history[j],neural)
            average_board_value[j][1]+= Inboard.game.get_corner_value(Inboard.board_history[j],neural,Inboard.game.getValidMoves(Inboard.board_history[j],neural),Inboard.game.getValidMoves(Inboard.board_history[j],-neural))
            average_board_value[j][2]+= Inboard.game.get_mobility_value(Inboard.board_history[j],neural,Inboard.game.getValidMoves(Inboard.board_history[j],neural),Inboard.game.getValidMoves(Inboard.board_history[j],-neural))
            average_board_value[j][3]+= Inboard.game.get_stability_value(Inboard.board_history[j],neural)

        
        for j in range(len(Inboard.action_history)):
            if Inboard.action_history[j][0]==neural:
                move = Inboard.game.action_to_move(Inboard.action_history[j][1])
                if move[0] != Inboard.game.n:
                    heatmap_neural[move[0]][move[1]]+=1

            if Inboard.prediction_history[j][0]==neural:
                evaluation_mcts[j+1][0] += 1
                evaluation_mcts[j+1][1] +=Inboard.prediction_history[j][1]
                evaluation_mcts[j+1][2] +=Inboard.prediction_history[j][2]
            if Inboard.prediction_history[j][0]==-neural:
                evaluation_minimax[j+1][0] += 1
                if Inboard.prediction_history[j][1] == math.inf:
                    evaluation_minimax[j+1][1] += 1
                elif Inboard.prediction_history[j][1] == -math.inf:
                    evaluation_minimax[j+1][1] += -1
                else:
                    evaluation_minimax[j+1][1] +=Inboard.prediction_history[j][1]
                       

                if Inboard.prediction_history[j][2]<10:
                    evaluation_minimax[j+1][2] +=Inboard.prediction_history[j][2]
                else:
                    evaluation_minimax[j+1][2] += 10

        for j in range(len(Inboard2.action_history)):
            if Inboard2.action_history[j][0]==neural:
                move = Inboard2.game.action_to_move(Inboard2.action_history[j][1])
                if move[0] != Inboard2.game.n:
                    heatmap_neural[move[0]][move[1]]+=1
        for j in range(len(Inboard3.action_history)):
            if Inboard3.action_history[j][0]==neural:
                move = Inboard3.game.action_to_move(Inboard3.action_history[j][1])
                if move[0] != Inboard3.game.n:
                    heatmap_neural[move[0]][move[1]]+=1

        a = Inboard.game.getGameEnded(Inboard.board_history[len(Inboard.board_history)-1],neural)
        if a ==1:
            wins+=1
        if a ==-1:
            losses+=1
        if a == -0.05:
            draws += 1


import seaborn as sns
sns.set_theme()
ax = sns.heatmap(heatmap_neural)
ax.figure.savefig("output.png")


print(evaluation_mcts)
print(str(wins) + " "+str(losses)+" "+str(draws))

"""

wandb.init(project="final evaluation")

for j in range(len(score_difference)):
    if score_difference[j][1]>0:
        if evaluation_mcts[j][0]>0 and evaluation_minimax[j][0]>0:
            wandb.log({"Average disk difference":score_difference[j][0]/score_difference[j][1],"turn":j,
                        "Neural prediction":evaluation_mcts[j][1]/evaluation_mcts[j][0],     
                        "MCTS sims":evaluation_mcts[j][2]/evaluation_mcts[j][0],
                        "Minimax prediction":evaluation_minimax[j][1]/evaluation_minimax[j][0],
                        "Depth searched":evaluation_minimax[j][2]/evaluation_minimax[j][0],
                        "Disc value":average_board_value[j][0]/score_difference[j][1],
                        "Corner value":average_board_value[j][1]/score_difference[j][1],
                        "Mobility value":average_board_value[j][2]/score_difference[j][1],
                        "stability value":average_board_value[j][3]/score_difference[j][1]})

#evaluation_mcts[j]/(score_difference[j][1]/2)
#evaluation_mcts[j]/(score_difference[j][1]/2)
"""



