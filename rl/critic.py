from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


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
    def update_value_function(self, state, td_error):
        pass

    @abstractmethod
    def update_eligibilities(self, state, discount_rate, decay_factor):
        pass


class TableBasedCritic(Critic):

    def __init__(self, learning_rate):
        # maps states to values
        self.state_values = dict()
        self.eligibilities = dict()
        self.learning_rate = learning_rate

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

    def update_value_function(self, state, td_error):
        """
        Updates the value of the given state based on td_error and the learning_rate
        """
        self.state_values[state] += self.learning_rate * td_error * self.eligibilities[state]

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

    def __init__(self, learning_rate, nn_dims):
        self.learning_rate = learning_rate
        self.nn_dims = nn_dims
        self.model = self.construct_nn()

    # not required
    def add_state(self, state):
        pass

    def construct_nn(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        for units in self.nn_dims:
            model.add(tf.keras.layers.Dense(units=units, activation=tf.nn.relu))
        model.compile(
            optimizer=optimizer,
            loss='mse'
        )
        return model

    # not required
    def reset_eligibilities(self):
        pass

    def compute_td_error(self, current_state, successor_state, reinforcement, discount_rate):
        return reinforcement + discount_rate * self.model.predict(successor_state) - self.model.predict(current_state)

    def update_value_function(self, state, td_error):
        # pred = self.model.predict(successor_state)
        # target = reinforcement + discount_rate * pred
        # self.model.train(current_state, target)
        pass

    # not required
    def update_eligibilities(self, state, discount_rate, decay_factor):
        pass
