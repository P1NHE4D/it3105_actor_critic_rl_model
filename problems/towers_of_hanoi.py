from rl.environment import Domain, State

# TODO question: 
# > For example, the 3-peg, 4-disc problem requires a minimum of 15 moves.
# does that mean we should support an initial state that is not cone shaped?

class TowersOfHanoi(Domain):

    def __init__(self, config):
        pass

    def produce_initial_state(self):
        pass

    def generate_child_state(self, state: State, action):
        pass

    def is_terminal_state(self, state: State):
        pass


class TowersOfHanoiState(State):

    def __init__(self, actions):
        super().__init__(actions)

