from __future__ import print_function
import sys
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from Game import Game
from othello.OthelloLogic import Board
import numpy as np
from time import perf_counter

class OthelloGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        dif = b.countDiff(player)
        if  dif > 0:
            return 1
        if dif == 0:
            return -0.05
        return -1

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):

        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    def action_to_move(self,action):
        if action !=None:
            move = (action%self.n,int(action/self.n))
            return move
        return None

    def move_to_action(self,move):
        if move != None:
            action = move[0]*self.n +move[1]
            return action
        return None 

    def get_num_corner_stones(self,player,board,legalMoves):
        score = 0
        corner_squares = ((0,0),(0,self.n-1),(self.n-1,0),(self.n-1,self.n-1))
        for x,y in corner_squares:
            if board[x][y]==player:
                score += 1
            
    """
    def get_corner_heuristic_value(self,maxplayer,board):
        max_corner_value = self.get_num_corner_stones(maxplayer,board)
        min_corner_value = self.get_num_corner_stones(maxplayer*-1,board)
        if(max_corner_value+min_corner_value)!=0:
            return 100*(max_corner_value)
        else:
            return 0
    """
    def get_only_pieces_of_player(self,board,player):
        new_board = board.copy()
        if player <0:
            new_board[new_board>0]=0
        else:
            new_board[new_board<0]=0
        return new_board

    def get_coin_parity(self,board,maxplayer):
        maxscore = np.sum(self.get_only_pieces_of_player(board,maxplayer))
        minscore = -1*np.sum(self.get_only_pieces_of_player(board,maxplayer*-1))
        return 100*(maxscore-minscore)/(maxscore+minscore)

    def get_mobility_score(self,board,maxplayer):
        maxpmv = len(self.getValidMoves(board,maxplayer))
        minpmv = len(self.getValidMoves(board,maxplayer*-1))

        if (maxpmv+minpmv)!=0:
            return 100*(maxpmv-minpmv)/(maxpmv+minpmv)
        else:
            return 0

    def get_static_weight_score(self,board,maxplayer):
        if self.n == 8:
            weights = np.array(([ 4,-3, 2, 2, 2, 2,-3, 4],
                                [-3,-4,-1,-1,-1,-1,-4,-3],
                                [ 2,-1, 1, 0, 0, 1, -1,2],
                                [ 2,-1, 0, 1, 1, 0,-1, 2],
                                [ 2,-1, 0, 1, 1, 0,-1, 2],
                                [ 2,-1, 1, 0, 0, 1,-1, 2],
                                [-3,-4,-1,-1,-1,-1,-4,-3],  
                                [ 4,-3, 2, 2, 2, 2,-3, 4]))
            return np.sum(weights*self.get_only_pieces_of_player(board,maxplayer)) + np.sum(weights*self.get_only_pieces_of_player(board,-maxplayer))


    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
