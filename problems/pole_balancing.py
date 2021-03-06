import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
from rl.env import Domain
from math import cos, sin
import numpy as np

ACTIONS = ["left", "right"]


@dataclass(frozen=True)
class Cart:
    velocity: float
    location: float
    angle: float
    angle_td: float

    def to_list(self):
        return [self.velocity, self.location, self.angle, self.angle_td]


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
        self.velocity_bins = compute_bins(*config["bins"]["velocity"])
        self.location_bins = compute_bins(*config["bins"]["location"])
        self.angle_bins = compute_bins(*config["bins"]["angle"])
        self.angle_td_bins = compute_bins(*config["bins"]["angle_td"])
        self.episode_count = 0
        self.step_count = []
        self.best_episode = []
        self.discretize = True
        self.states = []

    def get_init_state(self):
        self.states = []
        angle = np.random.uniform(-self.angle_magnitude, self.angle_magnitude)
        init_state = Cart(
            velocity=0,
            location=0,
            angle=angle,
            angle_td=0,
        )
        self.states.append(init_state)

        if self.discretize:
            angle = discretize_value(angle, self.angle_bins)

        return Cart(
            velocity=0,
            location=0,
            angle=angle,
            angle_td=0,
        ).to_list(), ACTIONS

    def get_child_state(self, action):
        state = self.states[-1]
        bb_force = -self.force if action == "left" else self.force

        angle_tdd = compute_angle_tdd(
            gravity=self.gravity,
            bb_force=bb_force,
            pole_mass=self.pole_mass,
            pole_length=self.pole_length,
            angle=state.angle,
            angle_td=state.angle_td,
            cart_mass=self.cart_mass
        )
        acceleration = compute_acceleration(
            bb_force=bb_force,
            pole_mass=self.pole_mass,
            pole_length=self.pole_length,
            cart_mass=self.cart_mass,
            angle=state.angle,
            angle_td=state.angle_td,
            angle_tdd=angle_tdd
        )
        angle_td = state.angle_td + self.timestep * angle_tdd
        velocity = state.velocity + self.timestep * acceleration
        angle = state.angle + self.timestep * angle_td
        location = state.location + self.timestep * velocity

        successor = Cart(
            velocity=velocity,
            location=location,
            angle=angle,
            angle_td=angle_td
        )
        self.states.append(successor)
        reinforcement = self.compute_reinforcement(successor)

        if self.discretize:
            angle = discretize_value(angle, self.angle_bins)
            angle_td = discretize_value(angle_td, self.angle_td_bins)
            velocity = discretize_value(velocity, self.velocity_bins)
            location = discretize_value(location, self.location_bins)

        return Cart(
            velocity=velocity,
            location=location,
            angle=angle,
            angle_td=angle_td
        ).to_list(), ACTIONS, reinforcement

    def is_current_state_terminal(self):
        state: Cart = self.states[-1]
        a = abs(state.angle) > self.angle_magnitude
        b = (state.location < self.left_boundary) or (state.location > self.right_boundary)
        c = len(self.states) > self.max_timesteps

        if a or b or c:
            self.step_count.append(len(self.states))
            self.episode_count += 1
            if len(self.states) >= len(self.best_episode):
                self.best_episode = self.states

        return a or b or c

    def visualise(self, actor):
        plt.plot(np.arange(1, self.episode_count + 1), self.step_count)
        plt.xlabel("Episode")
        plt.ylabel("Timestep")
        plt.show()
        plt.plot(np.arange(1, len(self.best_episode) + 1), list(map(lambda x: x.angle, self.best_episode)))
        plt.xlabel("Timestep")
        plt.ylabel("Angle")
        plt.show()

    def compute_reinforcement(self, state: Cart):
        angle_out_of_bounds = abs(state.angle) > self.angle_magnitude
        cart_out_of_bounds = state.location < self.left_boundary or state.location > self.right_boundary
        if angle_out_of_bounds or cart_out_of_bounds:
            return -100
        return 0


def compute_bins(start, stop, count):
    return np.linspace(start, stop, count)


def discretize_value(value, bins):
    return bins[min([np.digitize(value, bins), len(bins) - 1])]


def compute_angle_tdd(gravity, angle, bb_force, pole_mass, pole_length, angle_td, cart_mass):
    uu = cos(angle) * (-bb_force - pole_mass * pole_length * np.square(angle_td) * sin(angle))
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
