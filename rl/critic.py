from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from rl.utils import DefaultValueTable


class Critic(ABC):

    @abstractmethod
    def reset_eligibilities(self):
        pass

    @abstractmethod
    def increase_eligibility(self, state):
        pass

    @abstractmethod
    def compute_td_error(self, state, successor_state, reinforcement, discount_rate):
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
        self.state_values = DefaultValueTable(np.random.uniform)
        self.eligibilities = DefaultValueTable(lambda: 0)
        self.learning_rate = learning_rate

    def reset_eligibilities(self):
        """
        Resets all eligibilities to 0
        """
        for state_id in self.eligibilities.keys():
            self.eligibilities[state_id] = 0

    def increase_eligibility(self, state):
        state_id = hash(tuple(state))
        self.eligibilities[state_id] = 1

    def compute_td_error(self, state, successor_state, reinforcement, discount_rate):
        """
        computes the temporal difference error based on the reinforcement and value of a state
        :return: td error
        """
        state_id = hash(tuple(state))
        suc_state_id = hash(tuple(successor_state))
        return reinforcement + (discount_rate * self.state_values[suc_state_id]) - self.state_values[state_id]

    def update_value_function(self, state, td_error):
        """
        Updates the value of the given state based on td_error and the learning_rate
        """
        state_id = hash(tuple(state))
        self.state_values[state_id] += self.learning_rate * td_error * self.eligibilities[state_id]

    def update_eligibilities(self, state, discount_rate, decay_factor):
        """
        Updates the eligibility traces of the given state using the given discount rate and decay factor

        :param state: state for which the eligibility should be updated
        :param discount_rate: discount rate
        :param decay_factor: decay factor
        """
        state_id = hash(tuple(state))
        self.eligibilities[state_id] *= discount_rate * decay_factor

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

    # not required
    def increase_eligibility(self, state):
        pass

    def compute_td_error(self, state, successor_state, reinforcement, discount_rate):
        current_state = [state]
        successor_state = [successor_state]
        v_succ = self.model.predict(successor_state)[0, 0]
        v_curr = self.model.predict(current_state)[0, 0]
        return reinforcement + discount_rate * v_succ - v_curr

    def update_value_function(self, state, td_error):
        # pred = self.model.predict([successor_state])[0,0]
        # target = reinforcement + discount_rate * pred
        # self.model.fit([state], [target])
        pass

    # not required
    def update_eligibilities(self, state, discount_rate, decay_factor):
        pass
