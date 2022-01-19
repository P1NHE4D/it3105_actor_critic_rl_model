from rl.environment import Domain
import numpy as np

max_row = 4
max_col = 4
goal_row = 3
goal_col = 2
start_row = 0
start_col = 4


class Labyrinth(Domain):

    def __init__(self):
        self.states = {}

    def produce_initial_state(self):
        state = (start_row, start_col)
        if state.__hash__() not in self.states.keys():
            self.states[state.__hash__()] = state
        return state.__hash__()

    def generate_child_state(self, state_hash, action):
        state = self.states[state_hash]
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

        successor = (row, col)

        if successor.__hash__() not in self.states.keys():
            self.states[successor.__hash__()] = successor

        if row == goal_row and col == goal_col:
            reinforcement = 9
        else:
            reinforcement = -1
        return successor.__hash__(), reinforcement

    def is_terminal_state(self, state_hash):
        state = self.states[state_hash]
        return state[0] == goal_row and state[1] == goal_col

    def get_actions(self, state_hash):
        state = self.states[state_hash]
        row = state[0]
        col = state[1]
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

    def get_state(self, state_hash):
        return self.states[state_hash]

    def visualise_episode(self, episode):
        field = np.full((max_row + 1, max_col + 1), '#')
        for entry in episode:
            state = self.states[entry[0]]
            row = state[0]
            col = state[1]
            field[row, col] = '+'
        field[start_row, start_col] = 'o'
        field[goal_row, goal_col] = 'x'
        print()
        print("Total #steps needed: {}".format(len(episode)))
        print(field)
