from rl.environment import Domain, State


class PoleBalancing(Domain):

    def __init__(self, config):
        pass

    def produce_initial_state(self):
        pass

    def generate_child_state(self, state: State, action):
        pass

    def is_terminal_state(self, state: State):
        pass


class PoleBalancingState(State):

    def __init__(self):
        super().__init__(["positive_direction", "negative_direction"])

