from __future__ import print_function
import sys
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from Game import Game
from othello.OthelloLogic import Board
import numpy as np

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
        if b.countDiff(player) > 0:
            return 1
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
        if int(action/self.n)>=self.n:
            return "skip"
        move = (chr(97+int(action/self.n)), action%self.n)
        return move

######################################################################################################
    def get_better_value(self,player,board):
        max_player_moves = self.get_legal_moves(board,player)
        min_player_moves = self.get_legal_moves(board,-player)
        
        vc = self.get_coin_parity(board,player)
        vm = self.get_mobility_score(board,player,player_moves,opponent_moves)

    def get_corner_score(self,player,board,legalMoves):
        score = 0
        corner_squares = ((0,0),(0,self.n-1),(self.n-1,0),(self.n-1,self.n-1))
        for x,y in corner_squares:
            if board[x][y]==player:
                score += 1
            if (x,y) in legalMoves:
                score += 0.5

        
    def get_corner_value(self,maxplayer,board):
        max_corner_value = self.get_corner_score(maxplayer,board)
        min_corner_value = self.get_corner_score(maxplayer*-1,board)
        if(max_corner_value+min_corner_value)!=0:
            return 0.25*(max_corner_value-min_corner_value)/(max_corner_value+min_corner_value)
        return 0
    
    """
    def get_only_pieces_of_player(self,board,player):
        new_board = board.copy()
        if player <0:
            new_board[new_board>0]=0
        else:
            new_board[new_board<0]=0
        return new_board
    """

    def get_coin_value(self,board,maxplayer):
        maxscore = sum(board[board==maxplayer])
        minscore = -sum(board[board==-maxplayer])
        return (maxscore-minscore)/(maxscore+minscore)

    def get_mobility_value(self,board,player,player_moves,opponent_moves):
        pamv = len(player_moves[player_moves==1])
        oamv = len(opponent_moves[opponent_moves==1])

        empty_neighbours = []
        for pos in np.argwhere(board==-player):
            empty_neighbours += self.get_empty_neighbours(board,pos)
            unique_empty_neighbours = set(empty_neighbours)
        ppmv = len(unique_empty_neighbours)
        empty_neighbours = []
        for pos in np.argwhere(board==player):
            empty_neighbours += self.get_empty_neighbours(board,pos)
            unique_empty_neighbours = set(empty_neighbours)
        opmv = len(unique_empty_neighbours)
    
        if  ((pamv +0.5*ppmv) + (oamv +0.5*opmv))!=0:
            value = ((pamv +0.5*ppmv) - (oamv +0.5*opmv)) / ((pamv +0.5*ppmv) + (oamv +0.5*opmv))
        else:
            value = 0
        return value

    def get_empty_neighbours(self,board,position):
        empty_neighbours = []
        for i in range(-1,2):
            for j in range(-1,2):
                if not(i==position[0] and j ==position[1]):
                    if position[0]+i>=0 and position[0]+i<self.n and position[1]+j>=0 and position[1]+j<self.n:
                        if board[position[0]+i][position[1]+j] == 0:
                            empty_neighbours.append(((position[0]+i),(position[1]+j)))
        return empty_neighbours


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
            #return (np.sum(weights*self.get_only_pieces_of_player(board,maxplayer)) + np.sum(weights*self.get_only_pieces_of_player(board,-maxplayer)))/112
            return sum(board*weights) / 112
            # 112 is the maximum difference possible with 56 for player 1 and -56 for player 2
        if self.n == 6:
            weights = np.array(([ 5,-3, 3, 3,-3, 5],
                                [-3,-4,-1,-1,-4,-3],
                                [ 3,-1, 1, 1,-1, 3],
                                [ 3,-1, 1, 1,-1, 3],
                                [-3,-4,-1,-1,-4,-3],  
                                [ 5,-3, 3, 3,-3, 5]))
            #return (np.sum(weights*self.get_only_pieces_of_player(board,maxplayer)) + np.sum(weights*self.get_only_pieces_of_player(board,-maxplayer)))/96

            # 96 is the maximum difference possible with 48 for player 1 and -48 for player 2
"""
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
"""