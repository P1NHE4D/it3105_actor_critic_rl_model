import math
import numpy as np


class TableBasedActor:
    # contains policy, which computes a score expressing how desirable an action is in a given state

    def __init__(self):
        # maps state-action pairs to desirability value
        self.policy = dict()
        self.eligibilities = dict()
        self.state_actions = {}

    def add_state(self, state, actions):
        """
        Adds the given state to the state-action dictionary

        :param state: state to be added
        :param actions: possible actions in given state
        """
        self.state_actions[state] = actions
        for action in actions:
            if (state, action) not in self.policy.keys():
                self.policy[(state, action)] = 0
            if (state, action) not in self.eligibilities.keys():
                self.eligibilities[(state, action)] = 0

    def reset_eligibilities(self):
        """
        Resets the eligibility for every state-action pair to 0
        """
        for state_action in self.eligibilities:
            self.eligibilities[state_action] = 0

    def propose_action(self, state, epsilon):
        """
        proposes an action in a given state based on the desirability determined by the policy
        :param state: state object for which an action should be selected
        :param
        :param epsilon: probability for selecting a random action
        :return: an action
        """
        actions = self.state_actions[state]
        if np.random.choice(np.array([0, 1]), p=[1 - epsilon, epsilon]) == 1:
            return np.random.choice(np.array(actions))
        best_action = None
        max_value = -math.inf
        for action in actions:
            state_value = self.policy[(state, action)] / len(actions)
            if state_value > max_value:
                best_action = action
                max_value = state_value
        return best_action

    def update_policy(self, state, action, learning_rate, td_error):
        """
        Updates the policy using the td error computed by the critic

        :param state: state for which the policy should be updated
        :param action: corresponding action of the state
        :param learning_rate: learning rate
        :param td_error: temporal difference error computed by the critic
        """
        self.policy[(state, action)] += learning_rate * td_error * self.eligibilities[(state, action)]

    def update_eligibilities(self, state, action, discount_rate, decay_factor):
        """
        Updates the eligibilities for the given state-action pair based on the discount rate and
        decay factor.

        :param state: state for which the eligibility should be updated
        :param action: corresponding action of the state
        :param discount_rate: discount rate
        :param decay_factor: decay factor of eligibility
        """
        self.eligibilities[(state, action)] *= discount_rate * decay_factor
