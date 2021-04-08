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
        if action == None:
            return None
        if int(action/self.n)>=self.n:
            return "skip"
        move = (chr(97+int(action/self.n)), action%self.n)
        return move

######################################################################################################
    def get_better_value(self,board,player):
        player_moves = self.getValidMoves(board,player)
        opponent_moves = self.getValidMoves(board,-player)
        
        vcoin = self.get_coin_value(board,player)
        vmob  = self.get_mobility_value(board,player,player_moves,opponent_moves)
        vcorner = self.get_corner_value(board,player,player_moves,opponent_moves)
        vstability = self.get_stability_value(board,player)

        #print("coin: "+str(vcoin)+" mobility: "+str(vmob)+ " corner: "+ str(vcorner) + " stability: "+ str(vstability))

        return 0.25*vcoin+0.25*vmob+0.25*vcorner+0.25*vstability

    def get_corner_score(self,board,player,legalMoves):
        score = 0
        corner_squares = ((0,0),(0,self.n-1),(self.n-1,0),(self.n-1,self.n-1))
        for x,y in corner_squares:
            if board[x][y]==player:
                score += 1
            if (x,y) in legalMoves:
                score += 0.5
        return score

    def get_corner_value(self,board,player,player_moves,opponent_moves):

        player_corner_score = self.get_corner_score(board,player,player_moves)
        opponent_corner_score = self.get_corner_score(board,player*-1,opponent_moves)
        if(player_corner_score+opponent_corner_score)!=0:
            return (player_corner_score-opponent_corner_score)/(player_corner_score+opponent_corner_score)
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

    def get_coin_value(self,board,player):
        player_score = sum(board[board==player])
        opponent_score = -sum(board[board==-player])
        return (player_score-opponent_score)/(player_score+opponent_score)

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

            
    def get_edge_stability_matrix(self,board,player):
        n = self.n
        corners = ((0,0),(n-1,0),(n-1,n-1),(0,n-1))
        upper_edge = []
        lower_edge = []
        left_edge = []
        right_edge = []
        for i in range(1,n-1):
            upper_edge.append((i,0))
            right_edge.append((n-1,i))
            lower_edge.append((i,n-1))
            left_edge.append((0,i))
            
        edges = [upper_edge,right_edge,lower_edge,left_edge]
        alignment = [(1,0),(0,1),(1,0),(0,1)]
        stability_matrix = np.zeros((n,n))
        for corner in corners:
            if board[corner[0]][corner[1]]!=0:
                stability_matrix[corner[0]][corner[1]] = 1
        i = 0
        for edge in edges:
            #print("new edge")
            temp_list = edge+ [corners[i%4]] +[corners[(i+1)%4]]
            #print(temp_list)
            full_row = True
            for square in (temp_list):
                if board[square[0]][square[1]]==0:
                    full_row = False
                    break
                
            if full_row:
                for square in (temp_list):
                    #print("a")
                    stability_matrix[square[0]][square[1]]=1
            unchanged = True
            while unchanged == True:
                #print("durchlauf")
                unchanged = False
                for field in edge:
                    prev_field = (field[0]-alignment[i][0],field[1]-alignment[i][1])
                    if stability_matrix[field[0]][field[1]]==0:
                        if board[prev_field[0]][prev_field[1]]!=0 and board[prev_field[0]][prev_field[1]] == board[field[0]][field[1]]:
                            if stability_matrix[prev_field[0]][prev_field[1]]==1:
                                stability_matrix[field[0]][field[1]] = 1
                                unchanged = True
                                #print("set to 1")
                    #print(prev_field)
                #print("a")
                for field in edge:
                    prev_field = (field[0]+alignment[i][0],field[1]+alignment[i][1])
                    if stability_matrix[field[0]][field[1]]==0:
                        if board[prev_field[0]][prev_field[1]]!=0 and board[prev_field[0]][prev_field[1]] == board[field[0]][field[1]]:
                            if stability_matrix[prev_field[0]][prev_field[1]]==1:
                                stability_matrix[field[0]][field[1]] = 1
                                unchanged = True
                                #print("set to 1")
                #print(prev_field)
            i +=1
        return stability_matrix

    def get_stability_value(self,board,player):
        edge_stability_matrix = self.get_edge_stability_matrix(board,player)
        stable_coins = board*edge_stability_matrix
        player_stable_coins = sum(stable_coins[stable_coins==1])
        opponent_stable_coins = -sum(stable_coins[stable_coins==-1])

        if(player_stable_coins+opponent_stable_coins)!=0:
            return (player_stable_coins-opponent_stable_coins)/(player_stable_coins+opponent_stable_coins)
        return 0


    def get_static_weight_score(self,board,maxplayer):
        if self.n == 8:
            weights = np.array(([ 4,-3, 2, 2, 2, 2,-3, 4],
                                [-3,-4,-1,-1,-1,-1,-4,-3],
                                [ 2,-1, 1, 0, 0, 1,-1, 2],
                                [ 2,-1, 0, 1, 1, 0,-1, 2],
                                [ 2,-1, 0, 1, 1, 0,-1, 2],
                                [ 2,-1, 1, 0, 0, 1,-1, 2],
                                [-3,-4,-1,-1,-1,-1,-4,-3],  
                                [ 4,-3, 2, 2, 2, 2,-3, 4]))
            if maxplayer==1:
                return np.sum(weights*board)/112
            return -np.sum(weights*board)/112
            # 112 is the maximum difference possible with 56 for player 1 and -56 for player 2
        if self.n == 6:
            weights = np.array(([ 5,-3, 3, 3,-3, 5],
                                [-3,-4,-1,-1,-4,-3],
                                [ 3,-1, 1, 1,-1, 3],
                                [ 3,-1, 1, 1,-1, 3],
                                [-3,-4,-1,-1,-4,-3],  
                                [ 5,-3, 3, 3,-3, 5]))
            if maxplayer==1:
                return np.sum(weights*board)/96
            return -np.sum(weights*board)/96

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