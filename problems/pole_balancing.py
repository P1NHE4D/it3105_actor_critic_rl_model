from dataclasses import dataclass
import matplotlib.pyplot as plt
from rl.environment import Domain
from math import cos, sin
import numpy as np

ACTIONS = ["left", "right"]


@dataclass(frozen=True)
class Cart:
    velocity: float
    location: float
    angle: float
    angle_td: float
    angle_tdd: float


class PoleBalancing(Domain):

    def __init__(self, config):
        self.cart_mass = config["cart_mass"]
        self.pole_mass = config["pole_mass"]
        self.pole_length = config["pole_length"]
        self.force = config["force"]
        self.gravity = config["gravity"]
        self.angle_magnitude = config["angle_magnitude"]
        self.timestep = config["timestep"]
        self.left_boundary = config["left_boundary"]
        self.right_boundary = config["right_boundary"]
        self.max_timesteps = config["max_timesteps"]
        self.velocity_bins = np.linspace(*config["bins"]["velocity"])
        self.location_bins = np.linspace(*config["bins"]["location"])
        self.angle_bins = np.linspace(*config["bins"]["angle"])
        self.angle_td_bins = np.linspace(*config["bins"]["angle_td"])
        self.angle_tdd_bins = np.linspace(*config["bins"]["angle_tdd"])
        self.states: [Cart] = []
        self.episode_count = 0
        self.step_count = []

    def produce_initial_state(self):
        self.states = []
        angle = discretize_value(np.random.uniform(-self.angle_magnitude, self.angle_magnitude), self.angle_bins)
        init_state = Cart(
            velocity=0,
            location=0,
            angle=angle,
            angle_td=0,
            angle_tdd=0
        )
        self.states.append(init_state)
        return init_state.__hash__(), ACTIONS

    def generate_child_state(self, action):
        state = self.states[-1]
        bb_force = -self.force if action == "left" else self.force

        angle_tdd = compute_angle_tdd(
            gravity=self.gravity,
            bb_force=bb_force,
            pole_mass=self.pole_mass,
            pole_length=self.pole_length,
            angle=state.angle,
            angle_tdd=state.angle_tdd,
            cart_mass=self.cart_mass
        )
        acceleration = compute_acceleration(
            bb_force=bb_force,
            pole_mass=self.pole_mass,
            pole_length=self.pole_length,
            cart_mass=self.cart_mass,
            angle=state.angle,
            angle_td=state.angle_td,
            angle_tdd=state.angle_tdd
        )
        angle_td = state.angle_td + self.timestep * angle_tdd
        velocity = state.velocity + self.timestep * acceleration
        angle = state.angle + self.timestep * angle_td
        location = state.location + self.timestep * velocity

        successor = Cart(
            angle=discretize_value(angle, self.angle_bins),
            angle_td=discretize_value(angle_td, self.angle_td_bins),
            angle_tdd=discretize_value(angle_tdd, self.angle_tdd_bins),
            velocity=discretize_value(velocity, self.velocity_bins),
            location=discretize_value(location, self.location_bins)
        )
        self.states.append(successor)

        reinforcement = self.compute_reinforcement(successor)

        return successor.__hash__(), ACTIONS, reinforcement

    def is_current_state_terminal(self):
        state: Cart = self.states[-1]
        a = abs(state.angle) > self.angle_magnitude
        b = (state.location < self.left_boundary) or (state.location > self.right_boundary)
        c = len(self.states) > self.max_timesteps

        if a or b or c:
            self.step_count.append(len(self.states))
            self.episode_count += 1

        return a or b or c

    def visualise(self):
        plt.plot(np.arange(1, self.episode_count + 1), self.step_count)
        plt.xlabel("Episodes")
        plt.ylabel("Timesteps")
        plt.show()

    def compute_reinforcement(self, state: Cart):
        if abs(state.angle) > self.angle_magnitude:
            return -100
        if state.location < self.left_boundary or state.location > self.right_boundary:
            return -100
        if len(self.states) > self.max_timesteps:
            return 100
        return 1


def discretize_value(value, bins):
    return bins[min([np.digitize(value, bins), len(bins) - 1])]


def compute_angle_tdd(gravity, angle, bb_force, pole_mass, pole_length, angle_tdd, cart_mass):
    uu = cos(angle) * (-bb_force - pole_mass * pole_length * angle_tdd * sin(angle))
    ub = pole_mass + cart_mass
    u = gravity * sin(angle) + (uu / ub)
    bu = pole_mass * cos(angle) ** 2
    bb = pole_mass + cart_mass
    b = pole_length * ((4 / 3) - (bu / bb))
    return u / b


def compute_acceleration(bb_force, pole_mass, cart_mass, pole_length, angle, angle_td, angle_tdd):
    u = bb_force + pole_mass * pole_length * (angle_td ** 2 * sin(angle) - angle_tdd * cos(angle))
    b = pole_mass + cart_mass
    return u / b
