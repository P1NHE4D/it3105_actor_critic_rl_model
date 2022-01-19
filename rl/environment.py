from abc import ABC, abstractmethod


class Domain(ABC):

    @abstractmethod
    def produce_initial_state(self):
        pass

    @abstractmethod
    def generate_child_state(self, state_id, action):
        pass

    @abstractmethod
    def is_terminal_state(self, state_id):
        pass

    @abstractmethod
    def get_actions(self, state_id):
        pass

    @abstractmethod
    def visualise_episode(self, episode):
        pass
