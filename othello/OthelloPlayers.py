import numpy as np
import math
from time import perf_counter
from .OthelloGame import OthelloGame
from MCTS import MCTS



class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanOthelloPlayer():
    def __init__(self, game):
        self.game = game
        self.name = "Human"

    def play(self, board,deterministic=True):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x,y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
                            ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y if x != -1 else self.game.n ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a

    def reset(self):
        return


class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game
        self.name = "Greedy"

    def reset(self):
        return

    def play(self, board,deterministic=True):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        
        candidates.sort()
        rand_options = []
        best_indices = [i for i,x in enumerate(candidates) if x[0]==candidates[0][0]]
        random_best_indice = np.random.choice(best_indices)
        return candidates[random_best_indice][1]

class MinimaxOthelloPlayer():
    def __init__(self,game,depth,mode=1,maxtime=0):
        self.maxtime = maxtime
        self.game = game
        self.depth = depth
        self.alpha = -math.inf
        self.beta = math.inf
        self.name = "Minimax"
        self.mode = mode

    def play(self,board,deterministic=True,return_depth = False):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        if self.depth == 0:
            print("depth has to be atleast 1")

        if self.maxtime == 0:
            for a in range(self.game.getActionSize()):
                if valids[a]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board, 1, a)
                score = self.minimax(nextBoard,-1,self.depth,self.alpha,self.beta)
                candidates += [(-score, a)]
        else:
            start = perf_counter()
            endtime = start + self.maxtime
            for i in range(0,self.depth):
                
                temp_candidates = []
                for a in range(self.game.getActionSize()):
                    if valids[a]==0:
                        continue
                    nextBoard, _ = self.game.getNextState(board, 1, a)
                    score = self.minimax(nextBoard,-1,i,self.alpha,self.beta,endtime=endtime)
                    temp_candidates += [(-score, a)]
                if perf_counter() < endtime:
                    candidates = temp_candidates
                    
                else:
                    print("depth: "+str(i))
                    break
        
        candidates.sort()
        best_indices = [i for i,x in enumerate(candidates) if x[0]==candidates[0][0]]
        random_best_indice = np.random.choice(best_indices)
        if return_depth == True:
            return (candidates[random_best_indice][1],)
        return candidates[random_best_indice][1]


    def reset(self):
        return
    
    def minimax(self,board,player,depth,alpha,beta,endtime=0):

        if endtime != 0:
            if perf_counter()>endtime:
                return 0
        valids = self.game.getValidMoves(board,player)
        gameEnd = self.game.getGameEnded(board,player)
        if depth == 0 or gameEnd!=0:
            if depth == 0:
                if self.mode==1:
                    return self.game.getScore(board,1)
                if self.mode==2:
                    return self.game.get_mobility_score(board,1)+self.game.get_coin_parity(board,1)
                return self.game.get_static_weight_score(board,1)
            if gameEnd == 1:
                return math.inf
            if gameEnd == -1:
                return -math.inf
        if player == 1:
            score = -math.inf
            for a in range(self.game.getActionSize()):
                if valids[a]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board,1,a)
                score = max(score,self.minimax(nextBoard,-player,depth-1,alpha,beta,endtime=endtime))
                alpha = max(alpha,score)
                if alpha >= beta:
                    break
            return score
        if player == -1:
            score = math.inf
            for a in range(self.game.getActionSize()):
                if valids[a]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board,-1,a)
                score = min(score,self.minimax(nextBoard,-player,depth-1,alpha,beta,endtime=endtime))
                beta = min(beta,score)
                if beta <= alpha:
                    break
            return score

class NeuralNetworkPlayer():
    def __init__(self,game,nnet,args,name="NeuralPlayer"):
        self.game = game
        self.args = args
        self.nnet = nnet
        self.mctsStart = MCTS(game,nnet,args)
        self.mcts = MCTS(game,nnet,args)
        self.name = name

    def reset(self):
        self.mcts = self.mctsStart


    def play(self,board,tempThreshold_over=True,details = False):
        if tempThreshold_over:
            return np.argmax(self.mcts.getActionProb(board, temp=0))
        prob = self.mcts.getActionProb(board,temp=1)
        choice = np.random.choice(range(len(prob)),p=prob)

        return choice
        

    


    

        

    




