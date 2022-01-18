from abc import ABC, abstractmethod


class State(ABC):

    def __init__(self, actions):
        self.actions = actions


class Domain(ABC):

    @abstractmethod
    def produce_initial_state(self):
        pass

    @abstractmethod
    def generate_all_child_states_of_the_current_state(self, action):
        pass

    @abstractmethod
    def is_current_state_final_state(self, state: State):
        pass

    @abstractmethod
    def get_possible_actions(self):
        pass