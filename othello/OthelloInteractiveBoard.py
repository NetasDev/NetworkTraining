import pygame
import pygame.freetype
import os
import pickle
from .OthelloPlayers import *

class InteractiveBoard():
    def __init__(self,game,player1,player2):
        self.player1 = player1
        self.player2 = player2
        self.player1_name = self.player1.name
        self.player2_name = self.player2.name
        self.game = game
        self.player_to_move = 1
        self.size = self.game.n
        self.side_screen_size = 400
        self.screen_size = 1000
        self.square_size = 1000/self.size
        self.space = self.square_size/8
        self.board= self.game.getInitBoard()
        self.board_history = []
        self.action_history = []
        self.prediction_history = []
        self.move = 0

    def get_last_action(self,player):
        for i in range(len(self.action_history)):
            if self.action_history[len(self.action_history)-i-1][0] == player:
                return self.action_history[len(self.action_history)-i-1][1]
        return None
    
    def get_last_prediction(self,player):
        for i in range(len(self.prediction_history)):
            if self.prediction_history[len(self.prediction_history)-i-1][0] == player:
                return self.prediction_history[len(self.prediction_history)-i-1][1]
        return None

    def get_last_depth(self,player):
        for i in range(len(self.prediction_history)):
            if self.prediction_history[len(self.prediction_history)-i-1][0] == player:
                return self.prediction_history[len(self.prediction_history)-i-1][2]
        return None
    
    def human_players_turn(self):
        if(self.player_to_move == 1 and isinstance(self.player1,HumanOthelloPlayer)) or (self.player_to_move==-1 and isinstance(self.player2,HumanOthelloPlayer)):
            return True
            print("human move")
        return False

    def draw_field(self,screen):
        screen.fill((0,110,0))
        for row in range(self.size+2):
            pygame.draw.line(screen,(0,0,0),(row*self.square_size,0),(row*self.square_size,self.screen_size),width=4)

        #for col in range(self.size):
    
    def get_field_at_mouse_pos(self,pos):
        x,y = pos
        row = int(y /self.square_size)
        col = int(x /self.square_size)
        return row,col

    def draw_side_board(self,screen):
        Game_Font = pygame.freetype.Font(None,24)
        pygame.draw.rect(screen,(255,255,255),(self.screen_size,0,self.side_screen_size,self.screen_size))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size,150),"Player 1:",(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size,200),self.player1_name,(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size,250),"Last Move:",(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size+150,250),str(self.game.action_to_move(self.get_last_action(1))),(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size,300),"Score:",(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size+100,300),str(self.game.getScore(self.board,1)),(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size+100,350),str(self.get_last_prediction(1))[:6],(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size+100,400),str(self.get_last_depth(1)),(0,0,0))


        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size,550),"Player 2:",(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size,600),self.player2_name,(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size,650),"Last Move:",(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size+150,650),str(self.game.action_to_move(self.get_last_action(-1))),(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size,700),"Score:",(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size+150,700),str(self.game.getScore(self.board,-1)),(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size+150,750),str(self.get_last_prediction(-1))[:8],(0,0,0))
        Game_Font.render_to(screen,(self.screen_size+1/4*self.side_screen_size+100,800),str(self.get_last_depth(-1)),(0,0,0))
        

    def draw_board(self,screen):
        for i in range(self.size):
            for j in range(self.size):
                x = self.square_size*(i + 0.5)
                y = self.square_size*(j + 0.5)
                if self.board[i][j]==1:
                    pygame.draw.circle(screen,(0,0,0),(x,y),self.square_size/2-self.space/1.1)
                    pygame.draw.circle(screen,(0,0,0),(x,y),self.square_size/2-self.space)
                if self.board[i][j]==-1:
                    pygame.draw.circle(screen,(0,0,0),(x,y),self.square_size/2-self.space/1.1)
                    pygame.draw.circle(screen,(255,255,255),(x,y),self.square_size/2-self.space)
        if self.game.getGameEnded(self.board,self.player_to_move) == 0:
            moves = self.game.getValidMoves(self.board, self.player_to_move)
            for n in range(len(moves)):
                if moves[n]==1:
                    x = (int(n/self.size)+0.5)*self.square_size
                    y = ((n%self.size)+0.5)*self.square_size
                    if int(n/self.size) == self.game.n:
                        pygame.draw.circle(screen,(40,40,40),(x,y),self.square_size/2-self.space/1.1)
                        pygame.draw.circle(screen,(255,255,255),(x,y),self.square_size/2-self.space)
                        Game_Font = pygame.freetype.Font(None,24)
                        Game_Font.render_to(screen,(x-24,y-12),"Skip",(0,0,0))
                        break
                    pygame.draw.circle(screen,(0,0,0),(x,y),self.square_size/2-self.space/1.1)
                    pygame.draw.circle(screen,(0,110,0),(x,y),self.square_size/2-self.space)

            #for move in moves:

    def play_game(self):
        pygame.init()
        board = self.game.getInitBoard()
        self.board_history.append(self.board)
        self.player1.reset()
        self.player2.reset()

        #FPS = 60
        run = True
        #clock = pygame.time.Clock()
        screen = pygame.display.set_mode((self.screen_size+self.side_screen_size,self.screen_size))
        self.draw_field(screen)
        self.draw_side_board(screen)
        self.draw_board(screen)
        pygame.display.update()
        while run:
            #clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.human_players_turn():
                        pos = pygame.mouse.get_pos()
                        row,col = self.get_field_at_mouse_pos(pos)
                        moves = self.game.getValidMoves(self.board,self.player_to_move)
                        for i in range(len(moves)):
                            if moves[i]==1 and self.game.action_to_move(i)==(col,row):
                                self.board, self.player_to_move = self.game.getNextState(self.board,self.player_to_move,i)
                                self.board_history.append(self.board)
                                self.action_history.append((self.player_to_move*-1,self.game.move_to_action((col,row))))

            if not self.human_players_turn():
                if self.player_to_move ==1:
                    action = self.player1.play(self.game.getCanonicalForm(self.board,self.player_to_move))
                else:
                    action = self.player2.play(self.game.getCanonicalForm(self.board,self.player_to_move))
                self.board,self.player_to_move = self.game.getNextState(self.board,self.player_to_move,move)
                self.board_history.append(self.board)
                self.action_history.append((self.player_to_move*-1,action))

            self.draw_field(screen)
            self.draw_side_board(screen)
            self.draw_board(screen)

            if self.game.getGameEnded(self.board,self.player_to_move) !=0:
                break
            pygame.display.update()
        pygame.quit()
        return self.game.getGameEnded(self.board, 1)
    
    def save(self,path):
        self.player1 = None
        self.player2 = None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path+'.pkl','wb') as output:
            pickle.dump(self,output,pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path+'.pkl','rb') as input:
            return pickle.load(input)

    def show_replay(self):
        pygame.init()
        game = self.game
        board = self.board_history[0]
        move_his = self.action_history
        prediction_history = self.prediction_history 
        #InBoard = InteractiveBoard(board,game,len(board))
        #FPS = 60
        run = True
        #clock = pygame.time.Clock()
        screen = pygame.display.set_mode((self.screen_size+self.side_screen_size,self.screen_size))
        while run:
            #clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        if self.move > 0:
                            self.move = self.move -1
                            self.player_to_move *= -1
                        
                    if event.key == pygame.K_RIGHT:
                        if self.move < len(self.board_history)-1:
                            self.move = self.move +1
                            self.player_to_move *= -1
                    
            self.board = self.board_history[self.move]
            self.action_history = move_his[:self.move]
            self.prediction_history = prediction_history[:self.move]
            
            self.draw_field(screen)
            self.draw_side_board(screen)
            self.draw_board(screen)
            pygame.display.update()
        pygame.quit()

            
        

     