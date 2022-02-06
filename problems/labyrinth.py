from rl.env import Domain
import numpy as np

max_row = 4
max_col = 4
goal_row = 3
goal_col = 2
start_row = 0
start_col = 4


def get_actions(row, col):
    actions = []
    if row < max_row:
        actions.append("bottom")
    if row > 0:
        actions.append("top")
    if col < max_col:
        actions.append("right")
    if col > 0:
        actions.append("left")
    return actions


class Labyrinth(Domain):

    def __init__(self):
        self.states = []

    def get_init_state(self):
        self.states = []
        state = [start_row, start_col]
        self.states.append(state)
        actions = get_actions(start_row, start_col)
        return state, actions

    def get_child_state(self, action):
        state = self.states[-1]
        row = state[0]
        col = state[1]
        if action == "right":
            col = min(col + 1, max_col)
        elif action == "top":
            row = max(row - 1, 0)
        elif action == "bottom":
            row = min(row + 1, max_row)
        else:
            col = max(col - 1, 0)

        actions = get_actions(row, col)

        successor = [row, col]
        self.states.append(successor)

        reinforcement = -1
        return successor, actions, reinforcement

    def is_current_state_terminal(self):
        state = self.states[-1]
        return state[0] == goal_row and state[1] == goal_col

    def visualise(self):
        field = np.full((max_row + 1, max_col + 1), '#')
        for row, col in self.states:
            field[row, col] = '+'
        field[start_row, start_col] = 'o'
        field[goal_row, goal_col] = 'x'
        print()
        print("Total #steps needed: {}".format(len(self.states) - 1))
        print(field)
