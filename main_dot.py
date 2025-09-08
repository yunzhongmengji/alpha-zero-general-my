import logging, coloredlogs
from utils import dotdict
from Coach import Coach
from dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame as Game
from dotsandboxes.pytorch.NNet import NNetWrapper as nn

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

args = dotdict({
    'numIters': 8,                 # 训练轮数（别太小，给模型迭代的机会）
    'numEps': 16,                  # 每轮自博弈局数（样本量↑）
    'tempThreshold': 8,            # 前8步温度=1，局面更多样
    'updateThreshold': 0.58,       # 轻微放宽（从0.60→0.58）
    'maxlenOfQueue': 10000,        # 样本上限；避免过早截断
    'numMCTSSims': 40,             # MCTS模拟数；CPU还能承受
    'arenaCompare': 16,            # 竞技场对局数；评估更稳
    'cpuct': 1.25,                 # 略加强先验引导，防止搜过窄
    'checkpoint': './temp_min/',
    'load_model': False,
    'load_folder_file': ('./temp_min','best.pth.tar'),
    'numItersForTrainExamplesHistory': 10,  # 经验回放窗口
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
