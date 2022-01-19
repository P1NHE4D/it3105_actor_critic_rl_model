from abc import ABC, abstractmethod


class State(ABC):

    def __init__(self, actions):
        self.actions = actions

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class Domain(ABC):

    @abstractmethod
    def produce_initial_state(self):
        pass

    @abstractmethod
    def generate_child_state(self, state: State, action):
        pass

    @abstractmethod
    def is_terminal_state(self, state: State):
        pass
