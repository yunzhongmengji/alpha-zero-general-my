class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.
    这个类指定了游戏基类。如果要定义你自己的游戏，要继承这个类并实现以下的函数。
    这个设计适用于双人、对抗回合制的游戏。
    Use 1 for player1 and -1 for player2.
    使用 1 表示玩家 1，使用 -1 表示玩家 2
    See othello/OthelloGame.py for an example implementation.
    具体实现实例可参考othello/OthelloGame.py文件
    """
    def __init__(self):
        pass

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        返回游戏的初始棋盘
        返回值：
            初始棋盘：棋盘的表示形式（理想情况下，该形式应作为你的神经网络的输入
        """
        pass

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        返回棋盘大小
        返回值：
            (x,y)： 一个棋盘尺寸的数组
        """
        pass

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        返回动作空间大小
        返回值：
            动作空间大小：所有可能的行动数量、落子数量
        """
        pass

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player
        输入：
            棋盘：当前棋盘
            玩家：当前玩家（1 或者 -1）
            动作：当前玩家采取的动作

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        输出：
            下一个棋盘：应用动作后的棋盘
            下一个玩家：在下一个回合该谁行动的玩家
        """
        pass

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player
        输入：
            棋盘：当前棋盘
            玩家：当前玩家

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        输出：
            合法动作：一个长度为动作空间大小的二进制向量，用 1 来表示对于当前棋盘和当前玩家合法的动作，
                    用 0 表示对当前玩家和当前棋盘非法的动作
        """
        pass

    def getGameEnded(self, board, player):
        """
        判断游戏是否结束
        Input:
            board: current board
            player: current player (1 or -1)
        输入：
            棋盘：当前棋盘
            玩家：当前玩家

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
        输出：
            r:0 代表游戏没有结束，1代表当前玩家获胜，
                -1 代表当前玩家失败， 平局为接近0的小数
        """
        pass

    def getCanonicalForm(self, board, player):
        """
        返回以当前视角作为正方的棋盘，不用考虑对方怎么下，
        就是只用一个人的视角来看，只用学习一套策略
        Input:
            board: current board
            player: current player (1 or -1)
        输入：
            棋盘：当前棋盘
            玩家：当前玩家

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        输出：
            规范棋盘：返回棋盘的规范形式，这种规范形式应该独立于玩家，举个例子，在国际象棋中，
                    规范形式可以选择从白方的视角呈现，当前玩家是白方时，我们能直接返回当前棋盘，
                    当前玩家是黑方时，我们能反转棋子颜色后返回棋盘。
        """
        pass

    def getSymmetries(self, board, pi):
        """
        翻转棋盘，增加数据量
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()
        输入：
            棋盘:当前棋盘
            策略：动作空间大小的策略向量

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        输出：
            翻转形式：一个包含元组【（board，pi）】的列表，
            其中每个元组表示棋盘的翻转形态和对应的策略向量。
            这是用于神经网络训练的数据
        """
        pass

    def stringRepresentation(self, board):
        """
        将棋盘转换成字符串
        Input:
            board: current board
        输入：
            棋盘：当前棋盘

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        输出：
            棋盘字符串：将棋盘快速转换成字符串形式。
                    这种形式是MCTS进行哈希操作必须的。
        """
        pass
