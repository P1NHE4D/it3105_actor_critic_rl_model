from abc import ABC, abstractmethod


class State(ABC):

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def actions(self):
        pass


class Domain(ABC):

    def __init__(self, states):
        self.states = states

    @abstractmethod
    def produce_initial_state(self):
        pass

    @abstractmethod
    def generate_child_state(self, state: State, action):
        pass

    @abstractmethod
    def is_terminal_state(self, state: State):
        pass
