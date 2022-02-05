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
        self.discretize = config["discretize"]

    def fit(self, domain: Domain):
        """
        learns the target policy for a given domain

        :param domain: domain object for which the target policy should be learned
        """
        actor = TableBasedActor(self.actor_lr)
        if self.critic_type == "table":
            critic = TableBasedCritic(self.critic_lr)
        else:
            critic = NNBasedCritic(self.critic_lr, self.critic_nn_dims)

        # used for the progressbar only
        steps_per_episode = []

        progress = tqdm(range(self.max_episodes), desc="Episode", colour="green")
        for episode_count in progress:
            # TODO: add visualisation

            # reset eligibilities
            actor.reset_eligibilities()
            critic.reset_eligibilities()

            # get initial state and action
            current_state, actions = domain.get_init_state()
            actor.add_state(current_state, actions)
            critic.add_state(current_state)
            current_action = actor.propose_action(current_state, self.epsilon)

            # initialise an empty episode
            episode = []

            step = 0
            while step < self.steps and not domain.is_current_state_terminal():
                step += 1

                # append the current state-action pair to the current episode and initialise required values
                # in the actor and critic
                episode.append((current_state, current_action))

                # obtain a successor state and the reinforcement from moving to that state from the domain
                # TODO: should be an actual state, not a hash
                successor_state, actions, reinforcement = domain.get_child_state(current_action)

                # add successor states to actor and critic
                actor.add_state(successor_state, actions)
                critic.add_state(successor_state)

                # determine the best action from the successor based on the current policy
                successor_action = actor.propose_action(state=successor_state, epsilon=self.epsilon)
                # increase the eligibility of the current state
                actor.eligibilities[(current_state, current_action)] = 1
                # compute the td error using the current and the successor state
                td_error = critic.compute_td_error(
                    current_state=current_state,
                    successor_state=successor_state,
                    reinforcement=reinforcement,
                    discount_rate=self.discount_rate
                )
                critic.eligibilities[current_state] = 1

                # update the value function, eligibilities, and the policy for each state of the current episode
                for state, action in episode:
                    critic.update_value_function(state=state, td_error=td_error)
                    critic.update_eligibilities(state=state, discount_rate=self.discount_rate,
                                                decay_factor=self.decay_factor)
                    actor.update_policy(state=state, action=action, td_error=td_error)
                    actor.update_eligibilities(state=state, action=action, discount_rate=self.discount_rate,
                                               decay_factor=self.decay_factor)

                current_state = successor_state
                current_action = successor_action

            self.epsilon *= self.epsilon_decay
            if any(map(lambda x: x == episode_count, self.visualise)):
                domain.visualise()

            # update progressbar

            steps_per_episode.append(step)

            progress.set_description(
                f"[epsilon: {self.epsilon:.3f}] "
                f"[steps: (curr:{steps_per_episode[-1]} "
                f"min:{min(steps_per_episode)} "
                f"avg:{np.mean(steps_per_episode):.3f})] "
                f"[states: {critic.num_seen_states()}]"
            )

    def predict(self):
        pass
