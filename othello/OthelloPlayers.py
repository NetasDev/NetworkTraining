import numpy as np
import math
from time import perf_counter
from .OthelloGame import OthelloGame
from MCTS import MCTS



class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board,deterministic =True):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a
    
    def reset(self):
        return


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
            candidates += [(score, a)]
        
        candidates.sort(reverse=True)
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

    def play(self,board,deterministic=True,details = False):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        depth_searched = 0

        if self.maxtime == 0:
            depth_searched = self.depth
            for a in range(self.game.getActionSize()):
                if valids[a]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board, 1, a)
                score = self.minimax(nextBoard,-1,self.depth,self.alpha,self.beta)
                candidates += [(score, a)]
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
                    temp_candidates += [(score, a)]
                if perf_counter() < endtime:
                    candidates = temp_candidates
                    depth_searched = i
                else:
                    depth_searched = i-1
                    break
        
        candidates.sort(reverse=True)
        best_indices = [i for i,x in enumerate(candidates) if x[0]==candidates[0][0]]
        random_best_indice = np.random.choice(best_indices)
        if details == True:
            return (candidates[random_best_indice][1],candidates[random_best_indice][0],depth_searched)
        return candidates[random_best_indice][1]


    def reset(self):
        return
    
    def minimax(self,board,player,depth,alpha,beta,endtime=0):

        if endtime != 0:
            if perf_counter()>endtime:
                return 0
        valids = self.game.getValidMoves(board,player)
        gameEnd = self.game.getGameEnded(board,1)
        if depth == 0 or gameEnd!=0:
            if gameEnd == 1:
                return math.inf #win
            if gameEnd == -1:
                return -math.inf #loose
            if gameEnd !=0: #draw
                return 0

            if depth == 0:
                if self.mode==1:
                    return self.game.get_coin_value(board,1)
                if self.mode==2:
                    return self.game.get_better_value(board,1)
                return self.game.get_static_weight_score(board,1)
            
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
    def __init__(self,game,nnet,args,maxtime=0,name="NeuralPlayer"):
        self.game = game
        self.args = args
        self.nnet = nnet
        self.mctsStart = MCTS(game,nnet,args)
        self.mcts = MCTS(game,nnet,args)
        self.name = name
        self.maxtime = maxtime

    def reset(self):
        self.mcts = self.mctsStart


    def play(self,board,tempThreshold_over=True,details = False):
        if tempThreshold_over:
            if details:
                pi,Qs,simulations = self.mcts.getActionProb(board,temp=0,details=True,time=self.maxtime)
                a = np.argmax(pi)
                return a,float(Qs[a]),simulations
            else:
                return np.argmax(self.mcts.getActionProb(board, temp=0,time=self.maxtime))

        if details:
            pi,Qs,simulations = self.mcts.getActionProb(board,temp=0,details=True,time=self.maxtime)
            choice = np.random.choice(range(len(pi)),p=pi)
            return choice, float(Qs[choice]),simulations
        else:
                
            prob = self.mcts.getActionProb(board,temp=1,time=self.maxtime)
            choice = np.random.choice(range(len(prob)),p=prob)
            return choice
        

    


    

        

    




