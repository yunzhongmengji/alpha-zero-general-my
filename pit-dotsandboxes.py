import os
import numpy as np
import Arena
from MCTS import MCTS
from utils import dotdict
from dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame
from dotsandboxes.DotsAndBoxesPlayers import HumanDotsAndBoxesPlayer, RandomPlayer, GreedyRandomPlayer,StrongChainMCTSDotsAndBoxesPlayer,  HeuristicDotsAndBoxesPlayer
from dotsandboxes.pytorch.NNet import NNetWrapper

args = dotdict({
    'numIters': 50,                       # 多跑些大迭代，稳进
    'numEps': 20,                         # 12→30：自博弈量×2.5，数据更稳
    'tempThreshold': 15,                  # 前12步探索，数据更多样
    'updateThreshold': 0.6,              # 0.58→0.55：避免“差一点就不收录”
    'maxlenOfQueue': 30000,               # 4000→30000：别太快丢旧分布
    'numMCTSSims': 20  ,                    # 25→60：策略改进更有力（MX450还能扛）
    'arenaCompare': 40,                   # 10→40：评估方差显著下降
    'cpuct': 1.0,

    'checkpoint': './temp_min/',
    'load_model': True,
    'load_folder_file': ('./temp_min', 'best.pth.tar'),

    'numItersForTrainExamplesHistory': 10,# 3→20：缓解分布漂移导致的loss抬头
})


g = DotsAndBoxesGame(n=3)

p2 = StrongChainMCTSDotsAndBoxesPlayer(g).play
p3 = HeuristicDotsAndBoxesPlayer(g).play
p4 = RandomPlayer(g).play

numMCTSSims = 20
n1 = NNetWrapper(g)
n1.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
args1 = dotdict({'numMCTSSims': numMCTSSims, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
p1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

arena = Arena.Arena(p1, p3, g, display=DotsAndBoxesGame.display)
oneWon, twoWon, draws = arena.playGames(100, verbose=True)
print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))

# # Play Greedy vs Greedy
# p1 = grp1
# p2 = grp2
# arena = Arena.Arena(p1, p2, g, display=DotsAndBoxesGame.display)
# oneWon, twoWon, draws = arena.playGames(100, verbose=False)
# print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))

# # Play AlphaZero vs Greedy
# p1 = n1p
# p2 = grp2
# arena = Arena.Arena(p1, p2, g, display=DotsAndBoxesGame.display)
# oneWon, twoWon, draws = arena.playGames(2, verbose=False)
# print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))