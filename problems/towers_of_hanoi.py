from rl.env import Domain
from dataclasses import dataclass
from copy import deepcopy


# Notice that @dataclass means we'll get an __eq__ for free that is based on
# each property of the class ("eq=True" is the default). Notice also the use of
# "order=True"

@dataclass(order=True)
class Disk:
    """
    Disk is a single disk in the state description. It describes the "size" of
    the disk, such that if A and B are of type Disk, and A > B, then A is a
    bigger disk than B.
    """
    Size: int


@dataclass
class Peg:
    """
    Peg is a single peg in the state description. It describes the disks
    currently on that peg as a list. The first element is the top-most disk.
    """
    disks: list[Disk]

    def __hash__(self):
        # hash by converting to tuple
        hashed_disks = [hash(disk) for disk in self.disks]
        return hash(tuple(hashed_disks))


@dataclass
class State:
    """
    State captures a state of the problem an ordered list of pegs.
    """
    pegs: list[Peg]

    def __hash__(self):
        # hash by converting to tuple
        hashed_pegs = [hash(peg) for peg in self.pegs]
        return hash(tuple(hashed_pegs))


@dataclass
class Action:
    """
    Action can be applied to a state to produce a successor. "Pop peg index
    `popFrom` and push that value onto `pushTo`.
    """
    popFrom: int
    pushTo: int

    def __hash__(self):
        return hash(tuple([self.popFrom, self.pushTo]))


def is_success(state: State):
    """
    Returns True if the state describes a solved problem.
    """
    # success if all but the last pegs are empty. notice we don't have to check
    # for the order of the pegs on the last pole, since it is impossible to
    # achieve any other order than the correct one
    return all(len(peg.disks) == 0 for peg in state.pegs[:-1])


def legal_actions(state: State):
    """
    Generate legal actions in state.
    """
    actions = []

    # TODO this loop is an imperative mutating mess

    # for each peg, add an action to move it's top disk (if there is one) to
    # each other peg that has a smaller disk (or no disk)
    for idx_from, peg_from in enumerate(state.pegs):
        if len(peg_from.disks) == 0:
            # no disk to move
            continue
        for idx_to, peg_to in enumerate(state.pegs):
            if idx_from == idx_to:
                # can't move disk to own peg
                continue
            if len(peg_to.disks) == 0 or peg_from.disks[0] < peg_to.disks[0]:
                # moving disk is legal (it is larger, or target peg is empty)
                actions.append(Action(
                    popFrom=idx_from,
                    pushTo=idx_to,
                ))

    return actions


def successor(state: State, action: Action):
    """
    Generate successor of state through application of action. Assumes action is
    a legal one.
    """
    state = deepcopy(state)
    # pop front, mutating it in the process
    poppedDisk = state.pegs[action.popFrom].disks.pop(0)

    # yea yea I know, this way of adding something to a list prolly makes Python
    # sad. I want Python to be sad. Python makes me sad.
    state.pegs[action.pushTo].disks = [poppedDisk] + state.pegs[action.pushTo].disks

    return state


def calculate_reward(state_from, state_to):
    """
    Returns a reward associated with the state transition.
    """
    # uuh I created this function assuming I would reward success, but I think
    # it makes more sense to just penalize actions in general. Just return -1
    # regardless.
    return -1


class TowersOfHanoi(Domain):

    def __init__(self, num_pegs, num_disks):
        self.num_pegs = num_pegs
        self.num_disks = num_disks

        # to be set  by produce_initial_state
        self.states = None

    def get_current_state(self):
        return self.states[-1], legal_actions(self.states[-1])

    def produce_initial_state(self):
        # prepare initial state as one where all disks are on the first peg,
        # with smaller disks atop larger disks
        disks = list(range(self.num_disks))
        firstPeg = Peg(disks=disks)
        restPegs = [Peg(disks=[]) for _ in range(self.num_pegs - 1)]
        pegs = [firstPeg] + restPegs
        self.states = [State(pegs=pegs)]

        return self.get_current_state()

    def generate_child_state(self, action):
        self.states.append(
            successor(self.states[-1], action)
        )

        reward = calculate_reward(self.states[-2], self.states[-1])
        return *self.get_current_state(), reward

    def is_current_state_terminal(self):
        return is_success(self.states[-1])

    def visualise(self):
        # TODO matplotlib :)
        for i, state in enumerate(self.states):
            print(f"state {i}:")
            for peg in state.pegs:
                print(" " + str(peg.disks))
            print()
        print(f"solved the problem in {len(self.states) - 1} moves")
