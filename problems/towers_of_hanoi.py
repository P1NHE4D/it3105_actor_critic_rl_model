import time

import numpy as np
from matplotlib import pyplot as plt
from rl.env import Domain
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path


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

    def vector(self):
        return np.array([disk.Size for disk in self.disks])


@dataclass
class State:
    """
    State captures a state of the problem an ordered list of pegs.
    """
    pegs: list[Peg]

    def vector(self):
        total_disks = sum([len(peg.disks) for peg in self.pegs])
        total_pegs = len(self.pegs)

        # array of OHE peg placement of each disk, in increasing disk size
        # peg placement for peg i encoded as an N length array where the ith
        # element is 1, the rest is 0.
        # originally the order of disks on pegs was encoded as well, but through
        # https://www.cs.colostate.edu/~anderson/wp/pubs/Tower-of-Hanoi-1989.pdf
        # we realized this was not necessary, as any other order than smaller to larger is illegal
        ohe = np.zeros((total_disks * total_pegs))
        for curr_peg_i, peg in enumerate(self.pegs):
            for disk in peg.disks:
                size = disk.Size
                # calculate position in OHE where this disk's peg placement should be put
                ohe_disk_pos = size * total_pegs
                # encode peg placement
                ohe[ohe_disk_pos + curr_peg_i] = 1
        return ohe


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

    # for each peg, add an action to move its top disk (if there is one) to
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



class TowersOfHanoi(Domain):

    def __init__(
        self,
        num_pegs,
        num_disks,
        reward_success,
        reward_default,
        show_states_during_visualization=False,
        save_states_during_visualization=False,
    ):
        self.num_pegs = num_pegs
        self.num_disks = num_disks
        self.reward_success = reward_success
        self.reward_default = reward_default
        self.show_states_during_visualization = show_states_during_visualization
        self.save_states_during_visualization = save_states_during_visualization

        # to be set  by produce_initial_state
        self.states = []
        self.episode_count = 0
        self.state_counts = []
        self.avg_state_count = []

    def get_current_state(self):
        return self.states[-1].vector(), legal_actions(self.states[-1])

    def get_init_state(self):
        # for visualization purposes
        if len(self.states) > 0:
            self.episode_count += 1
            self.state_counts.append(len(self.states))
            self.avg_state_count.append(np.mean(self.state_counts))

        # prepare initial state as one where all disks are on the first peg,
        # with smaller disks atop larger disks

        disks = [Disk(Size=s) for s in list(range(self.num_disks))]
        firstPeg = Peg(disks=disks)
        restPegs = [Peg(disks=[]) for _ in range(self.num_pegs - 1)]
        pegs = [firstPeg] + restPegs
        self.states = [State(pegs=pegs)]

        return self.get_current_state()

    def get_child_state(self, action):
        self.states.append(
            successor(self.states[-1], action)
        )

        reward = self.calculate_reward(self.states[-2], self.states[-1])
        return *self.get_current_state(), reward

    def calculate_reward(self, state_from, state_to):
        """
        Returns a reward associated with the state transition.
        """
        if is_success(state_to):
            return (1 / len(self.states)) * self.reward_success
        return self.reward_default

    def is_current_state_terminal(self):
        return is_success(self.states[-1])

    def visualise(self, actor):
        plt.plot(np.arange(0, self.episode_count), self.state_counts)
        plt.plot(np.arange(0, self.episode_count), self.avg_state_count)
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.show()

        visualize_states(self.states, self.show_states_during_visualization, self.save_states_during_visualization)


def visualize_state(state: State):
    low, high = 0.2, 0.8
    peg_xs = np.linspace(low, high, len(state.pegs))
    peg_width = (high - low) / (len(state.pegs))
    num_disks = np.sum([len(peg.disks) for peg in state.pegs])
    height_ys = np.linspace(low, high, num_disks)

    fig, ax = plt.subplots()
    for i, peg in enumerate(state.pegs):
        for j, disk in enumerate(reversed(peg.disks)):
            disk_diameter = np.interp(
                disk.Size,
                [0, num_disks],
                [peg_width / 2, peg_width],
            )
            ax.add_patch(plt.Circle(
                (peg_xs[i], height_ys[j]),
                disk_diameter,
                color='g',
                clip_on=False
            ))

    return fig


def visualize_states(states: list[State], show=True, save=True):
    prefix = str(time.time())
    Path("./plots").mkdir(parents=True, exist_ok=True)
    for i, state in enumerate(states):
        fig = visualize_state(state)
        plt.title(f"STEP {i}")
        # save figure to disk
        if save:
            plt.savefig(f'./plots/{prefix}_step_{i}.png')

        if show:
            # user wants popup. do that
            plt.show()
        else:
            # force matplotlib to not show the figures..
            # this makes the learning process halt for around a second for some reason
            plt.close()

if __name__ == '__main__':
    # if ran directly, show that visualization of an episode works

    # let states be a sequence of states showing the optimal solution to the 3x3 problem
    states = [
        State(
            pegs=[
                Peg(disks=[
                    Disk(Size=0),
                    Disk(Size=1),
                    Disk(Size=2),
                ]),
                Peg(disks=[]),
                Peg(disks=[]),
            ]
        )
    ]
    states.append(successor(states[-1], Action(popFrom=0, pushTo=2)))
    states.append(successor(states[-1], Action(popFrom=0, pushTo=1)))
    states.append(successor(states[-1], Action(popFrom=2, pushTo=1)))
    states.append(successor(states[-1], Action(popFrom=0, pushTo=2)))
    states.append(successor(states[-1], Action(popFrom=1, pushTo=0)))
    states.append(successor(states[-1], Action(popFrom=1, pushTo=2)))
    states.append(successor(states[-1], Action(popFrom=0, pushTo=2)))

    visualize_states(states)
