from rl.environment import Domain, State


class Gambler(Domain):

    def __init__(self):
        states = []
        super().__init__(states)

    def produce_initial_state(self):
        pass

    def generate_child_state(self, state: State, action):
        pass

    def is_terminal_state(self, state: State):
        pass


class GamblerState(State):

    def __hash__(self):
        pass

    def actions(self):
        pass
