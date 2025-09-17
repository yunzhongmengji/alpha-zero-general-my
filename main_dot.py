import logging, coloredlogs
from utils import dotdict
from Coach import Coach
from dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame as Game
from dotsandboxes.pytorch.NNet import NNetWrapper as nn

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

args = dotdict({
    'numIters': 200,
    'numEps': 100,
    'tempThreshold': 8,
    'updateThreshold': 0.58,
    'maxlenOfQueue': 10000,
    'numMCTSSims': 200,
    'arenaCompare': 40,
    'cpuct': 1.25,
    'checkpoint': './temp_min/',
    'load_model': False,
    'load_folder_file': ('./temp_min','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
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
