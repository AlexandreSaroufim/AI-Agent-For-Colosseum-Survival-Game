# Student agent: Add your own agent here
from agents import random_agent
from agents.agent import Agent
from store import register_agent
import sys
import math


class Node:
    def __init__(self, state, parent):
        self.state = state
        self.wins = 0
        self.playouts = 0
        self.ucb = 0
        self.parent = parent
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

    # updates ucb then returns updated values
    def update_ucb(self, root):
        # can alter C to possibly improve ucb
        ucb1 = self.wins / self.playouts + math.sqrt(2) * math.sqrt(math.log(root.playouts, math.e))

        self.ucb = ucb1

        return self.ucb


class Monte_Carlo:

    # limit of 30 seconds to choose first move
    def __init__(self, state, length, max_moves):
        self.tree = Node(state, None)
        self.length = length
        self.max_moves = max_moves

        # limit to 28 seconds
        for i in range(1000):
            self.monte_carlo()

    def monte_carlo(self):
        leaf = self.select()
        child = self.expand(leaf)
        result = self.simulate(child)
        self.back_propagate(result, child)

    # Traverse Tree using children with highest UCB
    def select(self):
        cur_node = self.tree
        max_child_ucb = None

        while len(cur_node.children) != 0:
            for node in cur_node.children:
                if max_child_ucb is None or node.ucb > max_child_ucb.ucb:
                    max_child_ucb = node
            cur_node = max_child_ucb
            max_child_ucb = None

        return cur_node

    # Grow search tree by generating a new child of the selected leaf node
    # we can also generate several children
    def expand(self, leaf):
        # child based on leaf
        child = self.generate_child(leaf)
        leaf.add_child(child)
        return child

    # simulate a game using child as state
    # use random_agent using child.state
    def simulate(self, child):
        is_end, p0_score, p1_score = child.state.world.step()
        while not is_end:
            is_end, p0_score, p1_score = child.state.world.step()

    # Use result of the simulation to update all the search tree nodes going up to the root
    def back_propagate(self, result, child):
        if child.parent is None:
            return
        child.update_ucb()
        self.back_propagate(self, result, child.parent)


    # no more than 2 seconds to choose next move
    # returns the child that has the highest number of playouts for current state
    # current state for 1 second
    def actions(self, state):
        root = self.tree
        self.tree = self.find_node(root)

        # should last one second
        for i in range(10):
            self.monte_carlo()

        self.tree = root

        # return max ucb child of node with state

    # find node with specific state
    def find_node(self, state):
        cur_node = self.tree
        while len(cur_node.children) != 0:
            for node in cur_node.children:
                if node.state == state:
                    return node
        return None

    # generate a child based on the leaf node that doesn't exist in the tree
    def generate_child(self, leaf):
        r_agent= random_agent.RandomAgent.__init__()
        next_pos, dir = r_agent.step(leaf.state[0], leaf.state[1], leaf.state[2], self.max_moves)
        pass




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
        return my_pos, self.dir_map["u"]
