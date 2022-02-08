from dataclasses import dataclass
import numpy as np
from rl.env import Domain
import matplotlib.pyplot as plt


def get_actions(units):
    return np.arange(start=1, stop=min(units, 100 - units) + 1)


class Gambler(Domain):

    def __init__(self, config):
        self.win_prob = config.get("win_prob", 0.5)
        self.states = []
        self.episode_count = 0
        self.state_count = []

    def get_init_state(self):
        self.states = []
        self.episode_count += 1
        units = np.random.randint(1, 99+1)
        state = [units]
        self.states.append(state)
        actions = get_actions(units)
        one_hot_state = np.zeros(101)
        one_hot_state[units] = 1
        return one_hot_state, actions

    def get_child_state(self, action):
        outcome = np.random.choice([-action, action], p=[1-self.win_prob, self.win_prob])
        units = self.states[-1][0] + outcome
        state = [units]
        self.states.append(state)
        actions = get_actions(units)

        if units <= 0:
            reinforcement = -100
        elif units >= self.states[0][0]:
            reinforcement = 1
        else:
            reinforcement = -1

        one_hot_state = np.zeros(101)
        one_hot_state[units] = 1
        return one_hot_state, actions, reinforcement

    def is_current_state_terminal(self):
        units = self.states[-1][0]
        a = (units <= 0)
        b = (units >= 100)

        return a or b

    def visualise(self, actor):
        states = np.arange(1, 99 + 1)
        opt_val = []
        for state in states:
            s = np.zeros(101)
            s[state] = 1
            action = np.arange(start=1, stop=min(state, 100 - state) + 1)
            opt_val.append(actor.propose_action(s, action))
        plt.plot(states, opt_val)
        plt.show()
