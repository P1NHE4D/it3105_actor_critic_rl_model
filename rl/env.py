from abc import ABC, abstractmethod


class Domain(ABC):

    @abstractmethod
    def produce_initial_state(self):
        pass

    @abstractmethod
    def generate_child_state(self, action):
        pass

    @abstractmethod
    def is_current_state_terminal(self):
        pass

    @abstractmethod
    def visualise(self):
        pass
