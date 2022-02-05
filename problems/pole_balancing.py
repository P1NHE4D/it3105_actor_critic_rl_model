from rl.env import Domain
import numpy as np


class PoleBalancingState:

    def __init__(
            self,
            pole_angle,
            pole_angle_fst_td,
            pole_angle_sec_td,
            horiz_loc,
            horiz_vel,
            horiz_acc,
            timestep,
    ):
        super().__init__(["positive", "negative"])
        self.pole_angle = pole_angle
        self.pole_angle_fst_td = pole_angle_fst_td
        self.pole_angle_sec_td = pole_angle_sec_td
        self.horiz_loc = horiz_loc
        self.horiz_vel = horiz_vel
        self.horiz_acc = horiz_acc
        self.timestep = timestep


class PoleBalancing(Domain):

    def __init__(self, config):
        self.pole_length = config["pole_length"]
        self.pole_mass = config["pole_mass"]
        self.cart_mass = config["cart_mass"]
        self.gravity = config["gravity"]
        self.force = config["force"]
        self.pole_angle_magnitude = config["pole_angle_magnitude"]
        self.left_cart_bound = config["left_cart_bound"]
        self.right_cart_bound = config["right_cart_bound"]
        self.sim_timestep = config["sim_timestep"]
        self.max_timesteps = config["max_timesteps"]

    def get_init_state(self):
        return PoleBalancingState(
            pole_angle=np.random.uniform(-self.pole_angle_magnitude, self.pole_angle_magnitude, 1),
            pole_angle_fst_td=0,
            pole_angle_sec_td=0,
            horiz_loc=(self.right_cart_bound - self.left_cart_bound) / 2,
            horiz_vel=0,
            horiz_acc=0,
            timestep=1
        )

    def get_child_state(self, state: PoleBalancingState, action):
        # determining bang-bang force based on the selected action
        bb_force = self.force if action == "positive" else -self.force

        # second temporal derivative of the pole angle
        r1_tt = np.cos(state.pole_angle) * (-bb_force - self.pole_mass * self.pole_length * state.pole_angle_sec_td * np.sin(state.pole_angle))
        r1_tb = self.pole_mass + self.cart_mass
        r1_bt = self.pole_mass * np.square(np.cos(state.pole_angle))
        r1_bb = self.pole_mass + self.cart_mass
        pole_angle_sec_td = (self.gravity * np.sin(state.pole_angle) + (r1_tt/r1_tb)) / (self.pole_length * ((4 / 3) - (r1_bt / r1_bb)))

        # horizontal acceleration of the cart
        r2_t = bb_force + self.pole_mass * self.pole_length * (np.square(state.pole_angle_fst_td) * np.sin(state.pole_angle) - pole_angle_sec_td * np.cos(state.pole_angle))
        r2_b = self.pole_mass + self.cart_mass
        horiz_acc = r2_t / r2_b

        # first temporal derivative of the pole angle
        pole_angle_fst_td = state.pole_angle_fst_td + self.sim_timestep * pole_angle_sec_td

        # horizontal velocity of the cart
        horiz_vel = state.horiz_vel + self.sim_timestep * horiz_acc

        # pole angle
        pole_angle = state.pole_angle + self.sim_timestep * pole_angle_fst_td

        # horizontal location of the cart
        horiz_loc = state.horiz_loc + self.sim_timestep * horiz_vel

        # update timestep
        timestep = state.timestep + 1

        # compute reinforcement
        # TODO: needs to be optimised
        if np.abs(pole_angle) > self.pole_angle_magnitude and timestep < self.max_timesteps:
            reinforcement = -10
        elif timestep >= self.max_timesteps:
            reinforcement = 10
        else:
            reinforcement = timestep * 0.01

        return PoleBalancingState(
            pole_angle=pole_angle,
            pole_angle_fst_td=pole_angle_fst_td,
            pole_angle_sec_td=pole_angle_sec_td,
            horiz_loc=horiz_loc,
            horiz_vel=horiz_vel,
            horiz_acc=horiz_acc,
            timestep=timestep
        ), reinforcement

    def is_current_state_terminal(self, state: PoleBalancingState):
        return np.abs(state.pole_angle) > self.pole_angle_magnitude or state.timestep >= self.max_timesteps
