import math
import numpy as np
from matplotlib import pyplot as plt

from rl.utils import DefaultValueTable


class TableBasedActor:
    # contains policy, which computes a score expressing how desirable an action is in a given state

    def __init__(self, learning_rate, epsilon):
        # maps state-action pairs to desirability value
        self.policy = DefaultValueTable(lambda: 0)
        self.eligibilities = DefaultValueTable(lambda: 0)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def reset(self):
        """
        Resets the eligibility for every state-action pair to 0
        """
        for sa_id in self.eligibilities.keys():
            self.eligibilities[sa_id] = 0

    def increase_eligibility(self, state, action):
        state_id = hash(tuple(state))
        self.eligibilities[(state_id, action)] = 1

    def propose_action(self, state, actions):
        """
        proposes an action in a given state based on the desirability determined by the policy
        :param state: state object for which an action should be selected
        :param actions: actions that can be performed in the given state
        :param
        :return: an action
        """
        if len(actions) == 0:
            return None
        state_id = hash(tuple(state))
        if np.random.choice(np.array([0, 1]), p=[1 - self.epsilon, self.epsilon]) == 1:
            return np.random.choice(np.array(actions))
        best_action = None
        max_value = -math.inf
        for action in actions:
            state_value = self.policy[(state_id, action)] / len(actions)
            if state_value > max_value:
                best_action = action
                max_value = state_value
        return best_action

    def update_policy(self, episode, td_error):
        """
        Updates the policy using the td error computed by the critic

        :param episode:
        :param td_error: temporal difference error computed by the critic
        """
        for state, action in episode:
            state_id = hash(tuple(state))
            self.policy[(state_id, action)] += self.learning_rate * td_error * self.eligibilities[(state_id, action)]

    def update_eligibilities(self, episode, discount_rate, decay_factor):
        """
        Updates the eligibilities for the given state-action pair based on the discount rate and
        decay factor.

        :param episode:
        :param discount_rate: discount rate
        :param decay_factor: decay factor of eligibility
        """
        for state, action in episode:
            state_id = hash(tuple(state))
            self.eligibilities[(state_id, action)] *= discount_rate * decay_factor


    def visualize_policy(self):
        states = np.arange(1, 99+1)
        opt_val = []
        for state in states:
            s = np.zeros(101)
            s[state] = 1
            action = np.arange(start=1, stop=min(state, 100 - state) + 1)
            opt_val.append(self.propose_action(s, action))
        plt.plot(states, opt_val)
        plt.show()
