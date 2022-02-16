import numpy as np
from tqdm import tqdm

from rl.actor import TableBasedActor
from rl.critic import TableBasedCritic, NNBasedCritic
from rl.env import Domain


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

        # to be set by fit
        self.actor = None
        self.critic = None

    def fit(self, domain: Domain):
        """
        learns the target policy for a given domain

        :param domain: domain object for which the target policy should be learned
        """
        self.actor = TableBasedActor(
            learning_rate=self.actor_lr,
            epsilon=self.epsilon
        )
        if self.critic_type == "table":
            self.critic = TableBasedCritic(self.critic_lr)
        else:
            self.critic = NNBasedCritic(self.critic_lr, self.critic_nn_dims)

        # used for the progressbar only
        steps_per_episode = []

        progress = tqdm(range(self.max_episodes), desc="Episode", colour="green")
        for episode_count in progress:

            # reset actor and critic
            self.actor.reset()
            self.critic.reset()

            # get initial state and action
            current_state, actions = domain.get_init_state()
            current_action = self.actor.propose_action(current_state, actions)

            # initialise an empty episode
            episode = []

            step = 0
            while step < self.steps and not domain.is_current_state_terminal():
                step += 1

                # append the current state-action pair to the current episode and initialise required values
                # in the actor and critic
                episode.append((current_state, current_action))

                # obtain a successor state and the reinforcement from moving to that state from the domain
                successor_state, actions, reinforcement = domain.get_child_state(current_action)

                # determine the best action from the successor based on the current policy
                successor_action = self.actor.propose_action(state=successor_state, actions=actions)

                # increase the eligibility of the current state
                self.actor.increase_eligibility(current_state, current_action)

                # compute the td error using the current and the successor state
                td_error = self.critic.compute_td_error(
                    state=current_state,
                    successor_state=successor_state,
                    reinforcement=reinforcement,
                    discount_rate=self.discount_rate
                )
                self.critic.increase_eligibility(current_state)

                # update the value function, eligibilities, and the policy for each state of the current episode
                self.critic.update_value_function(episode=episode)
                self.critic.update_eligibilities(episode=episode, discount_rate=self.discount_rate, decay_factor=self.decay_factor)
                self.actor.update_policy(episode=episode, td_error=td_error)
                self.actor.update_eligibilities(episode=episode, discount_rate=self.discount_rate, decay_factor=self.decay_factor)

                current_state = successor_state
                current_action = successor_action

            self.epsilon *= self.epsilon_decay
            if any(map(lambda x: x == episode_count, self.visualise)):
                domain.visualise(self.actor)

            # update progressbar

            steps_per_episode.append(step)

            progress.set_description(
                f"[epsilon: {self.epsilon:.3f}] "
                f"[steps: (curr:{steps_per_episode[-1]} "
                f"min:{min(steps_per_episode)} "
                f"avg:{np.mean(steps_per_episode):.3f})] "
            )

    def predict(self, domain : Domain):
        # get initial state and action
        current_state, actions = domain.get_init_state()

        # temporarily set actor epsilon to 0
        original_epsilon = self.actor.epsilon
        self.actor.epsilon = 0

        steps = 0
        while steps < self.steps and not domain.is_current_state_terminal():
            steps += 1
            current_action = self.actor.propose_action(current_state, actions)
            current_state, actions, reward = domain.get_child_state(current_action)

        domain.visualise(self.actor)

        # revert epsilon to its original value
        self.actor.epsilon = original_epsilon
