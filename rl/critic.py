from abc import ABC, abstractmethod
import tensorflow as tf


class Critic(ABC):

    @abstractmethod
    def add_state(self, state):
        pass

    @abstractmethod
    def reset_eligibilities(self):
        pass

    @abstractmethod
    def compute_td_error(self, current_state, successor_state, reinforcement, discount_rate):
        pass

    @abstractmethod
    def update_value_function(self, state, learning_rate, td_error):
        pass

    @abstractmethod
    def update_eligibilities(self, state, discount_rate, decay_factor):
        pass


class TableBasedCritic(Critic):

    def __init__(self, ):
        # maps states to values
        self.state_values = dict()
        self.eligibilities = dict()

    def add_state(self, state):
        """
        Adds the given state to the state-value dict

        :param state: state to be added
        """
        if state not in self.state_values.keys():
            self.state_values[state] = np.random.uniform()
        if state not in self.eligibilities.keys():
            self.eligibilities[state] = 0

    def reset_eligibilities(self):
        """
        Resets all eligibilities to 0
        """
        for state in self.eligibilities:
            self.eligibilities[state] = 0

    def compute_td_error(self, current_state, successor_state, reinforcement, discount_rate):
        """
        computes the temporal difference error based on the reinforcement and value of a state
        :return: td error
        """
        return reinforcement + (discount_rate * self.state_values[successor_state]) - self.state_values[current_state]

    def update_value_function(self, state, learning_rate, td_error):
        """
        Updates the value of the given state based on td_error and the learning_rate

        :param state: state for which the value should be updated
        :param learning_rate: learning rate
        :param td_error: temporal difference error
        """
        self.state_values[state] += learning_rate * td_error * self.eligibilities[state]

    def update_eligibilities(self, state, discount_rate, decay_factor):
        """
        Updates the eligibility traces of the given state using the given discount rate and decay factor

        :param state: state for which the eligibility should be updated
        :param discount_rate: discount rate
        :param decay_factor: decay factor
        """
        self.eligibilities[state] *= discount_rate * decay_factor

    def num_seen_states(self):
        return len(self.state_values)


class NNBasedCritic(Critic):
    def add_state(self, state):
        pass

    def reset_eligibilities(self):
        pass

    def compute_td_error(self, current_state, successor_state, reinforcement, discount_rate):
        pass

    def update_value_function(self, state, learning_rate, td_error):
        pass

    def update_eligibilities(self, state, discount_rate, decay_factor):
        pass
