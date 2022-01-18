from rl.environment import Domain, State
import numpy as np
import math

# maze is implicitly a 3x3 grid, where the robot is never allowed to move into
# walls

MAZE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
]

GOAL = [len(MAZE)-1, len(MAZE[0])-1]

def candidate_moves():
    return [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

def create_is_legal_action_predicate(position):
    return lambda action : is_position_within_grid(apply_action(position, action))

def is_position_within_grid(position):
    [x, y] = position
    return 0 <= x < len(MAZE) and 0 <= y < len(MAZE[0]) and MAZE[x][y] != 1

def apply_action(position, action):
    return [
        position[0] + action[0],
        position[1] + action[1],
    ]

def available_actions_at_position(position):
    candidates = candidate_moves()
    return list(filter(create_is_legal_action_predicate(position), candidates))


class RobotMaze(Domain):

    def __init__(self, config, visualize=False):
        self.position_goal = GOAL
        self.position_robot = self.position_goal
        self.visualize = visualize
    
    def produce_current_state(self):
        state = tuple(self.position_robot)
        return state

    def visualize_current_state(self):
        for i, row in enumerate(MAZE):
            for j, cell in enumerate(row):
                if self.position_robot == [i, j]:
                    print("R", end=" ")
                elif self.position_goal == [i, j]:
                    print("G", end=" ")
                elif cell == 1:
                    print("W", end=" ")
                else:
                    print(".", end=" ")

            print()

    def produce_initial_state(self):
        while self.is_current_state_final_state() or not is_position_within_grid(self.position_robot):
            self.position_robot=[
                np.random.choice(len(MAZE)),
                np.random.choice(len(MAZE[0])),
            ]
        
        if self.visualize:
            self.visualize_current_state()

        return self.produce_current_state()

    def generate_all_child_states_of_the_current_state(self, action):

        if action in available_actions_at_position(self.position_robot):
            # legal action, let's do it
            self.position_robot[0] += action[0]
            self.position_robot[1] += action[1]

        if self.visualize:
            print("action chosen:", action)
            self.visualize_current_state()

        # moving is penalized
        action_reward = -10

        #if self.is_current_state_final_state():
        #    # moving to the final state is great!
        #    action_reward += 100

        return self.produce_current_state(), action_reward

    def is_current_state_final_state(self):
        return self.position_robot == self.position_goal
    
    def get_possible_actions(self):
        return candidate_moves()

if __name__ == '__main__':
    w = RobotMaze({})

    # stupid sequence showing how an agent could solve the problem...
    print(w.is_current_state_final_state())
    print(w.produce_initial_state())
    print(w.generate_all_child_states_of_the_current_state((1,0)))
    print(w.generate_all_child_states_of_the_current_state((1,0)))
    print(w.generate_all_child_states_of_the_current_state((0,1)))
    print(w.generate_all_child_states_of_the_current_state((0,1)))
    print(w.is_current_state_final_state())