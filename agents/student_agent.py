# Student agent: Add your own agent here
from agents.random_agent import RandomAgent
from agents.agent import Agent
from store import register_agent
import sys
import math
from copy import deepcopy
import numpy as np
import random
import time

class Node:
    def __init__(self, state, parent):
        self.state = list(deepcopy(state))
        self.wins = 0
        self.wins = 0
        self.playouts = 0
        self.ucb = 0
        self.parent = parent
        self.children: list[Node] = []

        if parent is None:
            self.isMaxPlayer = True
        else:
            self.isMaxPlayer = not parent.isMaxPlayer

    def add_child(self, obj):
        self.children.append(obj)

    # updates ucb then returns updated values
    def update_ucb(self, root):
        # can alter C to possibly improve ucb

        if self.playouts != 0 and root.playouts != 0:
            ucb1 = self.wins / self.playouts + math.sqrt(2) * math.sqrt(math.log(root.playouts, math.e) / self.playouts)
            self.ucb = ucb1

        return self.ucb


class Game:

    def __init__(self, node: Node, maxMoves):
        self.nodeSim = deepcopy(node)
        self.chessBoard = self.nodeSim.state[0]
        self.myTurn = node.isMaxPlayer
        self.maxMoves = maxMoves

    @classmethod
    def check_endgame(cls, state):

        """
                       Check if the game ends and compute the current score of the agents.

                       Returns
                       -------
                       is_endgame : bool
                           Whether the game ends.
                       outcome : int
                        Whether we win or enemy or tie
                       """

        chess_board, my_pos, enemy_pos = state
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        board_size = chess_board.shape[0]

        x,y = my_pos
        r, w = enemy_pos

        if list(chess_board[x][y]).count(True) == 4:
            return True, 1

        if list(chess_board[r][w]).count(True) == 4:
            return True, 0

        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                        moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(my_pos)
        p1_r = find(enemy_pos)
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, -1
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            return True, 0
        elif p0_score < p1_score:
            return True, 1
        else:
            player_win = -1  # Tie
            return True, 2
    def play(self):
        while True:

            if self.myTurn:
                fut_pos, dir = self.gameStep(self.chessBoard, self.nodeSim.state[1], self.nodeSim.state[2], self.maxMoves)
                self.chessBoard[fut_pos[0]][fut_pos[1]][dir] = True
                self.nodeSim.state[1] = deepcopy(fut_pos)
                self.myTurn = not self.myTurn

                is_endgame, output = Game.check_endgame((self.chessBoard, self.nodeSim.state[1], self.nodeSim.state[2]))

                if is_endgame and output == 0:
                    return True

            if not self.myTurn:
                fut_pos, dir = self.gameStep(self.chessBoard, self.nodeSim.state[2], self.nodeSim.state[1],
                                             self.maxMoves)
                self.chessBoard[fut_pos[0]][fut_pos[1]][dir] = True
                self.nodeSim.state[2] = deepcopy(fut_pos)
                self.myTurn = not self.myTurn

                is_endgame, output = Game.check_endgame((self.chessBoard, self.nodeSim.state[1], self.nodeSim.state[2]))

                if is_endgame and (output == 1 or output == 2):
                    return False




    def gameStep(self, chess_board, my_pos, adv_pos, max_step):

        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos

        # while chess_board[r, c, dir]:
        #     dir = np.random.randint(0, 4)

        return my_pos, dir



class Monte_Carlo:

    # limit of 30 seconds to choose first move
    def __init__(self, state, length, max_moves):
        self.tree = Node(list(state), None)
        self.length = length
        self.max_moves = max_moves

        # limit to 28 seconds
        t_end = time.time() + 5
        while time.time() < t_end:
            self.monte_carlo()

    def monte_carlo(self):

        leaf = self.select()
        child = self.expand(leaf)

        if not child is None:
            result = self.simulate(child)
            self.back_propagate(result, child)

    # Traverse Tree using children with highest UCB
    def select(self):
        cur_node = self.tree
        max_child_ucb: Node = None

        while len(cur_node.children) != 0:
            for node in cur_node.children:
                node.update_ucb(self.tree)
                if node.playouts == 0 or max_child_ucb is None or (node.ucb > max_child_ucb.ucb and max_child_ucb.playouts != 0):
                    max_child_ucb = node
            cur_node = max_child_ucb
            max_child_ucb = None

        return cur_node

    # Grow search tree by generating a new child of the selected leaf node
    # we can also generate several children
    def expand(self, leaf):

        chessboardLeaf = leaf.state[0]
        x,y = leaf.state[1]

        #Don't waste time expanding a suicidal leaf
        if Game.check_endgame(leaf.state) == 1:
            leaf.playouts+=1
            leaf.update_ucb(self.tree)
            return None

        randomChildren = []
        for i in range(self.max_moves * 2):
            randomChildren.append(self.generate_child(leaf))

        leaf.children = randomChildren
        # aggressiveChildren = []
        # futCurMovs = []
        # for i in range(self.max_moves * 2):
        #     aggChild = self.chooseChildToCornerEnemy(leaf)
        #     if aggChild is not None:
        #         aggressiveChildren.append(aggChild)
        #         futCurMovs.append(aggChild.state[1])
        #         leaf.children.remove(aggChild)
        #
        # leaf.children = aggressiveChildren
        #
        # print("Expanded from ", leaf.state[1], " to future curPos moves: ", futCurMovs)
        # x,y = leaf.state[1]
        # print("curPos walls: ", leaf.state[0][x][y])
        return leaf.children[0] if len(leaf.children) != 0 else None

    # simulate a game using child as state
    # use random_agent using child.state
    def simulate(self, child):
        """
        Take my pos
        Enemy pos
        Simulate a game
        See who wins
        return True if I win
        False if he wins
        use random step until someone wins
        """
        gameObj = Game(Node(child.state, None), maxMoves=self.max_moves)

        return gameObj.play()



    # Use result of the simulation to update all the search tree nodes going up to the root
    def back_propagate(self, result, child: Node):

        # if result is true then we won, if false we didn't win

        curNode = child
        while curNode is not None:
            curNode.playouts += 1
            if result:
                curNode.wins += 1
            curNode.update_ucb(self.tree)
            curNode = curNode.parent


    #To be used with Monte-Carlo
    #Chooses child with maximum playouts -> most potential
    def chooseChildWithMaxPlayouts(self, parent: Node):

        maxPlayoutChild: Node = None

        for child in parent.children:
            x, y = child.state[1]
            childWalls = child.state[0][x][y]
            numChildWalls = list(childWalls).count(True)

            is_end_game, result = Game.check_endgame(child.state)

            #if next move will guarantee us victory return child
            if result == 0:
                return child

            # Chooses the maxPlayout child that isn't a suicidal move
            if maxPlayoutChild is None or maxPlayoutChild.playouts < child.playouts and numChildWalls < 4 and result != 1:
                maxPlayoutChild = child

        return maxPlayoutChild

    def actions(self, state):

        if not np.array_equal(self.tree.state[0], state[0]) or self.tree.state[1] != state[1] or self.tree.state[2] != state[2]:
            self.tree = Node(state, None)

        #run monte-carlo for 1.5 seconds
        t_end = time.time() + 1.8
        while time.time() < t_end:
            self.monte_carlo()

        #We can alter the child picking stategy
        bestChild = self.chooseChildWithMaxPlayouts(self.tree)

        #Get next pos and next dir
        next_pos = bestChild.state[1]
        wallsNow = self.tree.state[0][next_pos[0]][next_pos[1]]
        wallsFuture = bestChild.state[0][next_pos[0]][next_pos[1]]
        dir = [i for i, (x, y) in enumerate(zip(wallsNow, wallsFuture)) if x != y][0]

        # discard all nodes that are maxChild's nodes because all the next moves will be children of maxChild
        self.tree = bestChild
        self.tree.parent = None
        return next_pos, dir

    # generate a child based on the leaf node that doesn't exist in the tree
    def generate_child(self, leaf: Node):
        return self.random_child(leaf, leaf.isMaxPlayer)

    @classmethod
    def distance2points(cls, my_pos, adv_pos):

        distance = math.ceil(math.sqrt((my_pos[0] - adv_pos[0]) ** 2 + (my_pos[1] - adv_pos[1]) ** 2))

        return distance

    @staticmethod
    def direction(my_pos, adv_pos):

        verticalDiff = adv_pos[0] - my_pos[0]
        horizontalDiff = adv_pos[1] - my_pos[1]

        if abs(verticalDiff) > abs(horizontalDiff):
            if verticalDiff > 0:
                return 2
            else:
                return 0

        else:

            if horizontalDiff > 0:
                return 1
            else:
                return 3

    # ifMaxPlayer random my_pos, if not randomize adv_pos
    def random_child(self, leaf: Node, isMaxPlayer):
        # Moves (Up, Right, Down, Left)

        max_step = self.max_moves
        ori_pos = None

        if isMaxPlayer:
            chess_board, my_pos, adv_pos = deepcopy(leaf.state)
            ori_pos = deepcopy(my_pos)

        else:
            chess_board, adv_pos, my_pos = deepcopy(leaf.state)
            ori_pos = deepcopy(adv_pos)

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)

        # Random Walk
        #print("random walk")
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r][c][dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier in direction of adversary
        dir = self.direction(my_pos,adv_pos)
        #dir = np.random.randint(0, 4)
        r, c = my_pos

        # temporary solution
        counter = 0
        while chess_board[r][c][dir]:
            dir = np.random.randint(0, 4)
            counter += 1
            if counter > 10:
                break

        if isMaxPlayer:
            leafCopy = Node(deepcopy([chess_board, my_pos, adv_pos]), leaf)
        else:
            leafCopy = Node(deepcopy([chess_board, adv_pos, my_pos]), leaf)
        leafCopy.state[0][r, c, dir] = True

        return leafCopy

        # ifMaxPlayer random my_pos, if not randomize adv_pos

    def superior_child(self, leaf: Node, isMaxPlayer):
        # Moves (Up, Right, Down, Left)

        max_step = self.max_moves
        ori_pos = None

        if isMaxPlayer:
            chess_board, my_pos, adv_pos = deepcopy(leaf.state)
            ori_pos = deepcopy(my_pos)

        else:
            chess_board, adv_pos, my_pos = deepcopy(leaf.state)
            ori_pos = deepcopy(adv_pos)

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)

        # Random Walk
        # print("random walk")
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            #futureWalls = list(chess_board[r][c])
            # Special Case enclosed by Adversary
            k = 0
            while 0 > r > self.length-1 or 0 > c > self.length-1 or chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 200:
                    print("Superior child is random")
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos

        # temporary solution
        counter = 0
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)
            counter += 1
            if counter > 10:
                break

        leafCopy = Node(deepcopy([chess_board, my_pos, adv_pos]), leaf)

        if not isMaxPlayer:
            leafCopy.state[1] = (r, c)
        else:
            leafCopy.state[2] = (r, c)
        leafCopy.state[0][r, c, dir] = True

        return leafCopy


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        self.monte_carlo = None
        self.origState = None

        print("Student Agent class initialized")

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # dummy return

        if self.monte_carlo is None:
            self.monte_carlo = Monte_Carlo((chess_board, my_pos, adv_pos), len(chess_board), max_moves=max_step)
            self.origState = list((chess_board, my_pos, adv_pos))

        # for i in range(30):
        #     print("Simulation number: ", i)
        #     arr = []
        #     for i in range(10):
        #         gameObj = Game(Node(self.origState, None), maxMoves=max_step)
        #         arr.append(gameObj.play())
        #     print("We won ", arr.count(True), " times")

        timeStart = time.time()

        action = self.monte_carlo.actions((chess_board, my_pos, adv_pos))

        timeEnd = time.time()

        print("Elapsed time: ", timeEnd - timeStart)

        return action
        # return my_pos, self.dir_map["u"]
