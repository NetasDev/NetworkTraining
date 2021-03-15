import logging
import pandas as pd
import numpy as np
import os
import wandb

from tqdm import tqdm
from othello.OthelloInteractiveBoard import InteractiveBoard

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, det_turns,display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.det_turns = 0

    def playGame(self, verbose=False,save=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        self.player1.reset()
        self.player2.reset()

        if save!= False:
            InBoard = InteractiveBoard(self.game,self.player1,self.player2)
            InBoard.board_history.append(InBoard.board)

        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            """
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            """
            action = players[curPlayer + 1].play(self.game.getCanonicalForm(board, curPlayer),it>=3)

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

            if save != False:
                InBoard.board_history.append(board)
                InBoard.move_history.append((curPlayer*-1,self.game.action_to_move(action)))
        """
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        """
        if save != False:
            i = 1
            while os.path.isfile(save+"game"+str(i)+".pkl"):
                i=i+1
            InBoard.save(save+"game"+str(i))
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False,Interactive=False,save = False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        i=1

        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            
            if Interactive == True:
                InBoard = InteractiveBoard(self.game,self.player1,self.player2)
                gameResult = InBoard.play_game()
                print(gameResult)
                if save != False:
                    InBoard.save(save+"game"+str(i))
            else:
                gameResult = self.playGame(verbose=verbose,save=save)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
            i+=1

        self.player1, self.player2 = self.player2, self.player1
        i = 1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            if Interactive == True:
                InBoard = InteractiveBoard(self.game,self.player1,self.player2)
                gameResult = InBoard.play_game()
                print(gameResult)
                if save != False:
                    InBoard.save(save+"game"+str(i))
            else:
                gameResult = self.playGame(verbose=verbose,save=save)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
            i+=1
        return oneWon, twoWon, draws

    @staticmethod
    def play_tournament(players,num_matches,game,det_turns,savefolder=False):
        wins = np.zeros((len(players),len(players)),dtype=int)
        draws = np.zeros((len(players),len(players)),dtype=int)
        names = []

        for i in range(len(players)):
            names.append(players[i].name)
            for j in range(len(players)):
                if i!=j:
                    if savefolder!= False:
                        save = savefolder +"/"+ players[i].name +"VS" +players[j].name+"/"
                    else:
                        save = False
                    arena = Arena(players[i],players[j],game,det_turns=det_turns)
                    wins[i][j],_,draws[i][j] = arena.playGames(num_matches,save=save)

        print(names)
        df = pd.DataFrame(wins,columns=names,index=names)
        df2 = pd.DataFrame(draws,columns=names,index=names)

        df.to_csv(r""+savefolder+"/wins.csv")
        df2.to_csv(r""+savefolder+"/draws.csv")

        print(df)
        print(df2)

        np.savetxt('wins',wins,delimiter = " ")
        np.savetxt('draws',draws,delimiter=" ")

    @staticmethod
    def play_one_against_many(player,players,num_matches,game,det_turns,savefolder=False):
        wins = np.zeros((len(players),1))
        draws = np.zeros((len(players),1))

        names = []
        for i in range(len(players)):
            names.append(players[i].name)
            if savefolder!= False:
                save = save = savefolder +"/"+ players[i].name
            else:
                save = False
            arena = Arena(player,players[i],game,det_turns=det_turns)
            wins[i],_,draws[i] = arena.playGames(num_matches,save=save)

        df = pd.DataFrame(wins,colums=player.name,index=names)
        df2 = pd.DataFrame(wins,colums=player.name,index=names)

        np.savetxt('wins',wins,delimiter = " ")
        np.savetxt('draws',draws,delimiter=" ")

    

