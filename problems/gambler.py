from rl.environment import Domain, State


class Gambler(Domain):

    def __init__(self, config):
        pass

    def produce_initial_state(self):
        pass

    def generate_child_state(self, state: State, action):
        pass

    def is_terminal_state(self, state: State):
        pass


class GamblerState(State):

    def __init__(self, actions):
        super().__init__(actions)
