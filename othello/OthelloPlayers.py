import numpy as np
import math
from MCTS import MCTS
from .OthelloGame import OthelloGame

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

    def play(self, board):
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


class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game

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

class MinimaxPlayer():
    def __init__(self,game,depth):
        self.game = game
        self.depth = depth
        self.alpha = -math.inf
        self.beta = math.inf

    def play(self,board,deterministic=True):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.minimax(nextBoard,-1,self.depth,self.alpha,self.beta)
            candidates += [(-score, a)]
        candidates.sort()
        rand_options = []
        best_indices = [i for i,x in enumerate(candidates) if x[0]==candidates[0][0]]
        random_best_indice = np.random.choice(best_indices)
        return candidates[random_best_indice][1]

    def reset(self):
        return
    
    def minimax(self,board,player,depth,alpha,beta):
        valids = self.game.getValidMoves(board,player)
        gameEnd = self.game.getGameEnded(board,player)
        if depth == 0 or gameEnd!=0:
            if depth == 0:
                return self.game.getScore(board,1)
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
                score = max(score,self.minimax(nextBoard,-player,depth-1,alpha,beta))
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
                score = min(score,self.minimax(nextBoard,-player,depth-1,alpha,beta))
                beta = min(beta,score)
                if beta <= alpha:
                    break
            return score

class NeuralNetworkPlayer():
    def __init__(self,game,nnet,args):
        self.game = game
        self.mctsStart = MCTS(game,nnet,args)
        self.mcts = MCTS(game,nnet,args)

    def reset(self):
        self.mcts = self.mctsStart


    def play(self,board,deterministic=True):
        if deterministic:
            return np.argmax(self.mcts.getActionProb(board, temp=0))

        prob = self.mcts.getActionProb(board,temp=1)
        choice = np.random.choice(range(len(prob)),p=prob)
        return choice
        

    


    

        

    




