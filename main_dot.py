import logging, coloredlogs
from utils import dotdict
from Coach import Coach
from dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame as Game
from dotsandboxes.pytorch.NNet import NNetWrapper as nn

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

args = dotdict({
    'numIters': 1000,                       # 多跑些大迭代，稳进
    'numEps': 100,                         # 12→30：自博弈量×2.5，数据更稳
    'tempThreshold': 15,                  # 前12步探索，数据更多样
    'updateThreshold': 0.6,              # 0.58→0.55：避免“差一点就不收录”
    'maxlenOfQueue': 200000,               # 4000→30000：别太快丢旧分布
    'numMCTSSims': 25  ,                    # 25→60：策略改进更有力（MX450还能扛）
    'arenaCompare': 40,                   # 10→40：评估方差显著下降
    'cpuct': 1.0,

    'checkpoint': './temp_min/',
    'load_model': True,
    'load_folder_file': ('./temp_min', 'best.pth.tar'),

    'numItersForTrainExamplesHistory': 20,# 3→20：缓解分布漂移导致的loss抬头
})






def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(n=3)  # 小棋盘：更快（棋盘尺寸/动作数都由 Game 提供）
    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)   # 按 NeuralNet 统一接口包装（train/predict/save/load）

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)  # 自博弈→训练→新旧网对战→是否接受新网

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

if __name__ == "__main__":
    main()
