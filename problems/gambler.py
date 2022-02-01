from dataclasses import dataclass
import numpy as np
from rl.environment import Domain
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class GamblerState:
    units: int

    def get_actions(self):
        return np.arange(start=1, stop=min(self.units, 100 - self.units) + 1)


class Gambler(Domain):

    def __init__(self, config):
        self.win_prob = config.get("win_prob", 0.5)
        self.states = []
        self.episode_count = 0
        self.state_count = []

    def produce_initial_state(self):
        self.states = []
        self.episode_count += 1
        units = np.random.random_integers(1, 99)
        state = GamblerState(units=units)
        self.states.append(state)
        actions = state.get_actions()
        return state.__hash__(), actions

    def generate_child_state(self, action):
        outcome = np.random.choice([-action, action], p=[1-self.win_prob, self.win_prob])
        units = self.states[-1].units + outcome
        state = GamblerState(units=units)
        self.states.append(state)
        actions = state.get_actions()

        if units <= 0:
            reinforcement = -100
        elif units >= self.states[-2].units:
            reinforcement = 1
        else:
            reinforcement = -1

        return state.__hash__(), actions, reinforcement

    def is_current_state_terminal(self):
        units = self.states[-1].units
        a = (units <= 0)
        b = (units >= 100)
        if a or b:
            self.state_count.append(len(self.states))

        return a or b

    def visualise(self):
        # TODO
        pass
