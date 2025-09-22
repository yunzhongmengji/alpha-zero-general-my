import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


# Will play at random, unless there's a chance to score a square
class GreedyRandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        previous_score = board[0, -1]
        for action in np.nonzero(valids)[0]:
            new_board, _ = self.game.getNextState(board, 1, action)
            new_score = new_board[0, -1]
            if new_score > previous_score:
                return action
        a = np.random.randint(self.game.getActionSize())
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanDotsAndBoxesPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        if board[2][-1] == 1:
            # We have to pass
            return self.game.getActionSize() - 1
        valids = self.game.getValidMoves(board, 1)
        while True:
            print("Valid moves: {}".format(np.where(valids == True)[0]))
            a = int(input())
            if valids[a]:
                return a
            print('Invalid move')



# ---------- 规则/启发式玩家 ----------
class HeuristicDotsAndBoxesPlayer:
    """
    一个简单而实用的启发式对手：
    1) 优先：任何能立刻得分的着法（补成第4边）；
    2) 其次：避免制造“三边格”的着法；
    3) 若全是三边格（不可避免的送分局面）：选择预计给对手最少连吃的着法（粗略估计）。
    """
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1).astype(bool)
        actions = np.nonzero(valids)[0]
        if board[2, -1] == 1:
            # 有“必须Pass”的标志位（上一步得分），遵从规则
            return self.game.getActionSize() - 1

        # 1) 能立刻得分的着法
        best_scoring = []
        cur_score = board[0, -1]
        for a in actions:
            new_b, _ = self.game.getNextState(board, 1, a)
            if new_b[0, -1] > cur_score:
                best_scoring.append(a)
        if best_scoring:
            return int(np.random.choice(best_scoring))

        # 2) 过滤掉会制造“三边格”的危险着法
        safe_moves = []
        for a in actions:
            if not self._creates_third_side(board, a):
                safe_moves.append(a)
        if safe_moves:
            return int(np.random.choice(safe_moves))

        # 3) 都是三边格：粗略估计给对手的“连吃长度”，选最小
        best_a, best_penalty = None, float("inf")
        for a in actions:
            penalty = self._estimate_chain_penalty(board, a)
            if penalty < best_penalty:
                best_penalty, best_a = penalty, a
        return int(best_a)

    # —— 辅助：判断 action 是否制造“三边格”
    def _creates_third_side(self, board, action):
        n = self.game.n
        # 将 action 映射到 (x,y) 与水平/垂直边
        horiz_count = n * (n + 1)
        is_horizontal = action < horiz_count
        if is_horizontal:
            x = action // n
            y = action % n
            # 本仓库的存储：水平边在 board[:n+1,:n]，竖边在 board[-n:,:]
            # 将该边补上后，落在它两侧的任一格若已有另外两边，则形成三边
            # 为轻量判断，我们直接数该边相邻格当前已铺设的边数
            return self._adjacent_box_has_two(board, is_horizontal=True, x=x, y=y)
        else:
            a2 = action - horiz_count
            x = a2 // (n + 1)
            y = a2 % (n + 1)
            return self._adjacent_box_has_two(board, is_horizontal=False, x=x, y=y)

    def _adjacent_box_has_two(self, board, is_horizontal, x, y):
        n = self.game.n
        def count_edges_of_box(i, j):
            # 以仓库的布局：格 (i,j) 的四条边分别是：
            #  上：board[:n+1,:n] at (i, j)
            #  下：board[:n+1,:n] at (i+1, j)
            #  左：board[-n:,:]   at (i, j)
            #  右：board[-n:,:]   at (i, j+1)
            top = board[i, j]
            bottom = board[i+1, j]
            left = board[n+1 + i, j]
            right = board[n+1 + i, j+1]
            return int(top) + int(bottom) + int(left) + int(right)

        # 与该边相邻的格最多两个，检查是否已有“两边已下”
        # 将 action 衍生到相邻格的索引并统计
        candidates = []
        if is_horizontal:
            # 一条水平边位于 (x, y) 与 (x, y) 的“上/下”两格
            if x > 0:
                candidates.append((x-1, y))
            if x < n:
                candidates.append((x, y))
        else:
            # 一条竖直边位于 (x, y) 与 (x, y-1) 的“左/右”两格
            if y > 0:
                candidates.append((x, y-1))
            if y < n:
                candidates.append((x, y))

        for (i, j) in candidates:
            if 0 <= i < n and 0 <= j < n:
                if count_edges_of_box(i, j) == 2:
                    return True
        return False

    def _estimate_chain_penalty(self, board, action):
        # 简化估计：落下该边后，统计周围被“第三边化”的格数量作为惩罚
        n = self.game.n
        penalty = 0
        horiz_count = n * (n + 1)
        is_horizontal = action < horiz_count
        if is_horizontal:
            x = action // n
            y = action % n
            # 检查两侧格在“落子后”是否会从1->2边
            penalty += int(self._would_become_third(board, True, x, y, up=True))
            penalty += int(self._would_become_third(board, True, x, y, up=False))
        else:
            a2 = action - horiz_count
            x = a2 // (n + 1)
            y = a2 % (n + 1)
            penalty += int(self._would_become_third(board, False, x, y, up=True))
            penalty += int(self._would_become_third(board, False, x, y, up=False))
        return penalty

    def _would_become_third(self, board, is_horizontal, x, y, up=True):
        # 判断相邻某个格（若存在）的当前边数是否为1，并因本步变为2
        n = self.game.n
        def count_edges_of_box(i, j):
            top = board[i, j]
            bottom = board[i+1, j]
            left = board[n+1 + i, j]
            right = board[n+1 + i, j+1]
            return int(top) + int(bottom) + int(left) + int(right)

        if is_horizontal:
            if up and x > 0:
                i, j = x-1, y
            elif (not up) and x < n:
                i, j = x, y
            else:
                return False
        else:
            if up and y > 0:
                i, j = x, y-1
            elif (not up) and y < n:
                i, j = x, y
            else:
                return False

        if 0 <= i < n and 0 <= j < n:
            c = count_edges_of_box(i, j)
            return (c == 1)
        return False


# ---------- 纯 UCT / 随机回合 MCTS 玩家 ----------
class PureMCTSDotsAndBoxesPlayer:
    """
    只用随机回合进行评估的 MCTS（无神经网络）。
    参数：
      - num_sims: 每步的模拟次数
      - c_puct : UCT 探索常数
    """
    def __init__(self, game, num_sims=400, c_puct=1.4):
        self.game = game
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.Q = defaultdict(float)   # Q(s,a)
        self.N = defaultdict(int)     # N(s,a)
        self.Ns = defaultdict(int)    # N(s)
        self.Ps = dict()              # 均匀先验/展开后缓存
        self.terminal = dict()        # 缓存终局

    def play(self, board):
        if board[2, -1] == 1:
            return self.game.getActionSize() - 1
        for _ in range(self.num_sims):
            self._simulate(np.copy(board), 1)
        s_key = self._key(board, 1)
        valids = self.game.getValidMoves(board, 1).astype(bool)
        acts = np.nonzero(valids)[0]
        # 选择访问次数最多的动作
        counts = [self.N[(s_key, a)] for a in acts]
        return int(acts[int(np.argmax(counts))])

    # —— MCTS 核心 —— #
    def _simulate(self, board, curPlayer):
        s_key = self._key(board, curPlayer)
        # 终局
        if s_key in self.terminal:
            return self.terminal[s_key]
        r = self.game.getGameEnded(board, curPlayer)
        if r != 0:
            self.terminal[s_key] = r
            return r

        valids = self.game.getValidMoves(board, 1).astype(bool)
        acts = np.nonzero(valids)[0]

        # 未展开：初始化均匀先验，做一次随机 rollout 作为估值
        if s_key not in self.Ps:
            self.Ps[s_key] = np.ones_like(valids, dtype=np.float32) / max(1, len(acts))
            v = self._rollout(board, curPlayer)
            self.Ns[s_key] += 1
            return v

        # 选择：UCT
        best, best_u = -1, -1e9
        sqrt_sum = math.sqrt(self.Ns[s_key] + 1e-8)
        for a in acts:
            q = self.Q[(s_key, a)]
            n = self.N[(s_key, a)]
            u = q + self.c_puct * self.Ps[s_key][a] * sqrt_sum / (1 + n)
            if u > best_u:
                best_u, best = u, a

        # 前进
        next_board, next_player = self.game.getNextState(board, curPlayer, best)
        v = self._simulate(next_board, next_player)

        # 回传（注意视角转换：返回的是“对当前玩家”的结果）
        self.Q[(s_key, best)] = (self.N[(s_key, best)] * self.Q[(s_key, best)] + v) / (self.N[(s_key, best)] + 1)
        self.N[(s_key, best)] += 1
        self.Ns[s_key] += 1
        return v

    def _rollout(self, board, curPlayer):
        # 随机走到终局，返回胜负（+1 当前玩家胜，-1负）
        player = curPlayer
        b = np.copy(board)
        while True:
            r = self.game.getGameEnded(b, player)
            if r != 0:
                return r
            valids = self.game.getValidMoves(b, 1).astype(bool)
            acts = np.nonzero(valids)[0]
            a = int(np.random.choice(acts))
            b, player = self.game.getNextState(b, player, a)

    def _key(self, board, curPlayer):
        # 用游戏提供的 canonical 视角 + bytes 作为哈希 key
        cano = self.game.getCanonicalForm(board, curPlayer)
        return cano.tobytes()

