import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from time import perf_counter
from othello.OthelloPlayers import *
import wandb

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            #self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                if r!=1 and r!= -1:
                    return [(x[0],x[2],r) for x in trainExamples]
                    
                else:
                    return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        start_time = perf_counter()
        time_this_iteration = start_time
        all_time_selfplay,all_time_training,all_time_validation,all_wins,all_looses,all_draws,all_games,generation  = 0,0,0,0,0,0,0,0

        for i in range(1, self.args.numIters + 1):
            start_time_selfplay = perf_counter()
            # start of all/selfplay
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = []

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            #end of selfplay/start of training network
            start_time_training = perf_counter()
            # training new network, keeping a copy of the old one

            if perf_counter() - start_time >self.args.maxtime:
                print(perf_counter()-start_time)
                break
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp')
            pmctsplayer = NeuralNetworkPlayer(self.game,self.pnet,self.args)

            loss,pi_loss,v_loss = self.nnet.train(trainExamples)
            nmctsplayer = NeuralNetworkPlayer(self.game,self.nnet,self.args)



            # end of selfplay/start of validation
            start_time_validation = perf_counter()
            log.info('PITTING AGAINST PREVIOUS VERSION')

            arena = Arena(pmctsplayer,nmctsplayer,self.game,self.args.tempThreshold)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)


            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins+0.45*draws) / (pwins + nwins + draws) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best')
                generation = generation + 1

            # end of Validation
            # wandb speicherung anfang
            end_time_validation = perf_counter()
            time_this_iteration = perf_counter() - start_time_selfplay
            all_time = perf_counter()-start_time

            selfplay_time_iteration = start_time_training - start_time_selfplay
            training_time_iteration = start_time_validation - start_time_training
            validation_time_iteration = end_time_validation- start_time_validation

            all_time_selfplay = all_time_selfplay + start_time_training - start_time_selfplay
            all_time_training = all_time_training + start_time_validation - start_time_training
            all_time_validation = all_time_validation + end_time_validation- start_time_validation

            games = pwins+nwins+draws
            all_wins = all_wins + nwins 
            all_looses = all_looses + pwins
            all_draws = all_draws + draws
            all_games = all_games + pwins + nwins + draws
            
            wandb.log({"Wins":nwins,"Losses":pwins,"Draws":draws,"Win-Rate":nwins/games,"Overall Win-Rate":all_wins/all_games,
            "Overall Draw-Rate":all_draws/all_games,"Time":all_time,"Selfplay-Time Iteration":selfplay_time_iteration,
            "Training-Time Iteration":training_time_iteration,"Validation-Time Iteration":validation_time_iteration,
            "Selfplay-Time Iteration %":selfplay_time_iteration*100/time_this_iteration,
            "Traing-Time Iteration %":training_time_iteration*100/time_this_iteration,
            "Validation-Time Iteration %":validation_time_iteration*100/time_this_iteration,
            "Selfplay-Time":all_time_selfplay,"Selfplay-Time %":all_time_selfplay*100/all_time,
            "Training-Time":all_time_training,"Training-Time %":all_time_training*100/all_time,
            "Validation-Time":all_time_validation,"Validation-Time %":all_time_validation*100/all_time,
            "Generation":generation,"loss":loss,"pi_loss":pi_loss,"v_loss":v_loss
            })
            #wandb speicherung ende

            
    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration)

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True


        