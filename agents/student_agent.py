# Student agent: Add your own agent here
from agents.random_agent import RandomAgent
from agents.agent import Agent
from store import register_agent
import sys
import math
from copy import deepcopy
import numpy as np
import random


class Node:
    def __init__(self, state, parent):
        #self.state: list[list[list], tuple[int, int], tuple[int, int]] = state
        self.state = list(state)
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
            ucb1 = self.wins / self.playouts + math.sqrt(2) * math.sqrt(math.log(root.playouts, math.e))
            self.ucb = ucb1

        return self.ucb


class Monte_Carlo:

    # limit of 30 seconds to choose first move
    def __init__(self, state, length, max_moves):
        self.tree = Node(list(state), None)
        self.length = length
        self.max_moves = max_moves

        #Temporary
        self.found = 0
        self.played = 0

        # limit to 28 seconds
        for i in range(10):
            self.monte_carlo()
            print("init: " + str(i))

    def monte_carlo(self):

        print("select")
        leaf = self.select()
        print("expand")
        child = self.expand(leaf)
        print("simulate")
        result = self.simulate(child)
        print("back-prop")
        self.back_propagate(result, child)

    # Traverse Tree using children with highest UCB
    def select(self):
        cur_node = self.tree
        max_child_ucb: Node = None

        print("\n num_children", len(cur_node.children))
        while len(cur_node.children) != 0:
            for node in cur_node.children:
                if node.playouts == 0 or max_child_ucb is None or node.ucb > max_child_ucb.ucb:
                    max_child_ucb = node
            cur_node = max_child_ucb
            max_child_ucb = None

        return cur_node

    # Grow search tree by generating a new child of the selected leaf node
    # we can also generate several children
    def expand(self, leaf):
        # child based on leaf

        child = None
        for i in range(30):
            child = self.generate_child(leaf)
            leaf.add_child(child)
        
        return child

    # simulate a game using child as state
    # use random_agent using child.state
    def simulate(self, child):
        return random.choice([True, False])

    @staticmethod
    def check_endgame(state):

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

    # Use result of the simulation to update all the search tree nodes going up to the root
    def back_propagate(self, result, child: Node):

        # if result is true then we won, if false we didn't win

        curNode = child
        while curNode != None:
            curNode.playouts += 1
            if result:
                curNode.wins += 1
            curNode.update_ucb(self.tree)
            curNode = curNode.parent


    def actions(self, state):

        # if state is anchild of root, run monte-carlo on that child as the new root
        foundChild = False
        for child in self.tree.children:
            if child.state == state:
                self.tree = child
                foundChild = True
                self.found+=1
                break

        if not foundChild:
            self.tree = Node(state, None)

        #run monte-carlo for 1.5 seconds
        for i in range(20):
            self.monte_carlo()

        self.played+=1

        print("Found ratio: ", self.found, self.played)
        curNode: Node = self.tree
        maxChild = curNode.children[0]

        for child in curNode.children:
            x,y = child.state[1]
            childWalls = child.state[0][x][y]

            #temp
            #child.playouts > maxChild.playouts use this when simulate implemented
            if list(childWalls).count(True) < 2 and x != self.length-1 and x != 0 and y != self.length-1 and y != 0:
                maxChild = child

        next_pos = maxChild.state[1]

        wallsNow = self.tree.state[0][next_pos[0]][next_pos[1]]
        wallsFuture = maxChild.state[0][next_pos[0]][next_pos[1]]

        # find the altered index for 2 arrays
        dir = [i for i, (x, y) in enumerate(zip(wallsNow, wallsFuture)) if x != y][0]

        # discard all nodes that are maxChild's nodes because all the next moves will be children of maxChild
        self.tree = maxChild

        self.tree.parent = None

        print("Distance to enemy: ", self.distance2points(next_pos,state[2]), "Max steps: ", self.max_moves)

        return next_pos, dir

    # generate a child based on the leaf node that doesn't exist in the tree
    def generate_child(self, leaf: Node):
        # temporary
        return self.random_child(leaf, leaf.isMaxPlayer)


    def distance2points(self,my_pos,adv_pos):

        distance = math.ceil(math.sqrt((my_pos[0] - adv_pos[0]) ** 2 + (my_pos[1] - adv_pos[1]) ** 2))

        return distance


    def printTree(self):
        #use bfs to print tree
        pass

            


    # ifMaxPlayer random my_pos, if not randomize adv_pos
    def random_child(self, leaf: Node, isMaxPlayer):
        # Moves (Up, Right, Down, Left)

        max_step = self.max_moves
        chess_board, my_pos, adv_pos = deepcopy(leaf.state)
        ori_pos = None

        if isMaxPlayer:
            ori_pos = deepcopy(my_pos)
        else:
            ori_pos = deepcopy(adv_pos)

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)

        # Random Walk
        print("random walk")
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

        # temporary solution
        counter = 0
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)
            counter += 1
            if counter > 10:
                break

        leafCopy = Node([chess_board, my_pos, adv_pos], leaf)

        if isMaxPlayer:
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

        return self.monte_carlo.actions((chess_board, my_pos, adv_pos))
        # return my_pos, self.dir_map["u"]
