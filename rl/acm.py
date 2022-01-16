import numpy as np
import matplotlib as plt
import math

from rl.environment import Domain


class ACM:
    def __init__(self, config):
        self.max_episodes = config["episodes"]
        self.steps = config["steps"]
        self.critic_type = config["critic_type"]
        self.critic_nn_dims = config["critic_nn_dims"]
        self.actor_lr = config["actor_lr"]
        self.critic_lr = config["critic_lr"]
        self.decay_factor = config["decay"]
        self.discount_rate = config["discount"]
        self.epsilon = config["epsilon"]
        self.epsilon_decay = config["epsilon_decay"]
        self.visualise = config["visualise"]
        self.verbose = config["verbose"]

    def fit(self, domain: Domain):
        """
        learns the target policy for a given domain

        :param domain: domain object for which the target policy should be learned
        """
        actor = TableBasedActor(domain.states)
        if self.critic_type == "table":
            critic = TableBasedCritic(domain.states)
        else:
            critic = NNBasedCritic()
        for episode_count in range(self.max_episodes):

            # reset eligibilities
            actor.reset_eligibilities()
            critic.reset_eligibilities()

            # get initial state
            current_state = domain.produce_initial_state()
            # get initial action
            current_action = actor.propose_action(current_state, self.epsilon)

            episode = []

            step = 0
            while step < self.steps and not domain.is_terminal_state(current_state):
                step += 1
                episode.append((current_state, current_action))
                successor_state, reinforcement = domain.generate_child_state(current_state, current_action)
                successor_action = actor.propose_action(successor_state)
                actor.eligibilities[(current_state.__hash__(), current_action)] = 1
                td_error = critic.compute_td_error()
                critic.eligibilities[current_state] = 1

                for state_action in episode:
                    critic.update_value_function()
                    critic.eligibilities[state_action[0]] = self.discount_rate * self.decay_factor * critic.eligibilities[state_action[0]]
                    actor.update_policy(td_error)
                    actor.eligibilities[state_action] = self.discount_rate * self.decay_factor * actor.eligibilities[
                        state_action]

                current_state = successor_state
                current_action = successor_action

            self.epsilon *= self.epsilon_decay

    def predict(self):
        pass


class TableBasedActor:
    # contains policy, which computes a score expressing how desirable an action is in a given state

    def __init__(self, states):
        # maps state-action pairs to desirability value
        self.policy = dict()
        self.eligibilities = dict()
        for state in states:
            for action in state.actions():
                self.policy[(state.__hash__(), action)] = 0
                self.eligibilities[(state.__hash__(), action)] = 0

    def reset_eligibilities(self):
        for state_action in self.eligibilities:
            self.eligibilities[state_action] = 0

    def propose_action(self, state, epsilon):
        """
        proposes an action in a given state based on the desirability determined by the policy
        :param state: state object for which an action should be selected
        :param epsilon: probability for selecting a random action
        :return: an action
        """
        if np.random.choice(np.array([0, 1]), p=[1 - epsilon, epsilon]) == 1:
            return np.random.choice(np.array(state.actions()))
        best_action = None
        max_value = -math.inf
        for action in state.actions:
            state_value = self.policy[state.__hash__(), action]
            if state_value > max_value:
                best_action = action
                max_value = state_value
        return best_action

    def update_policy(self, td_error):
        """
        Updates the policy using the td error computed by the critic
        :param td_error: temporal difference error computed by the critic
        """
        # TODO: normalise values of each state across the number of actions that can be taken in a given state
        pass

    def update_eligibilities(self):
        pass


class TableBasedCritic:

    def __init__(self, states):
        # maps states to values
        self.state_values = dict()
        self.eligibilities = dict()
        for state in states:
            self.state_values[state.__hash__()] = np.random.uniform()
            self.eligibilities[state.__hash__()] = 0

    def reset_eligibilities(self):
        for state in self.eligibilities:
            self.eligibilities[state] = 0

    def compute_td_error(self):
        """
        computes the temporal difference error based on the reinforcement and value of a state
        :return: td error
        """
        pass

    def update_value_function(self):
        pass

    def update_eligibilities(self):
        pass


class NNBasedCritic:
    pass
