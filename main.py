import logging

import coloredlogs
import wandb

from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.keras.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps':  100,             # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.55,    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/Othello8x8/LearningRate/01/',
    'load_model': False,
    'load_folder_file': ('./temp/Othello8x8/FirstModel/','best'),
    'numItersForTrainExamplesHistory': 10,
    'maxtime': 10800,

    'wandb_project':'Othello8x8 - learning rate'
})

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(8)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        #c.loadTrainExamples()

    run = wandb.init(project=args.wandb_project,config=args,reinit=True)

    log.info('Starting the learning process 🎉')
    c.learn()
    run.finish()



main()
