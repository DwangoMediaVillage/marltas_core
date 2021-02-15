"""CNN-DQN actor."""
from collections import deque
from typing import Callable, List, Optional

import gym
import numpy as np

from dqn.actor import ActorBase
from dqn.cnn.config import CNNConfigBase
from dqn.cnn.datum import SampleFromActor
from dqn.cnn.policy import Policy
from dqn.explorer import Explorer
from dqn.make_env import EnvConfig, make_env
from dqn.policy import PolicyParam
from dqn.utils import np_inverse_value_scaling, np_value_scaling


class Actor(ActorBase):
    """Actor for sampling experience using CNN-QNetwork.

    Args:
        init_policy_param: Initial policy parameters.
        config: CNN configuration.
        process_index: Index pf process.
        make_env_func: Function to initialize an environment.
    """
    def __init__(self,
                 init_policy_param: PolicyParam,
                 config: CNNConfigBase,
                 process_index: int = 0,
                 make_env_func: Callable[[EnvConfig, int, int], gym.Env] = make_env):
        super(Actor, self).__init__(vector_env_size=config.actor.vector_env_size,
                                    process_index=process_index,
                                    env_config=config.env,
                                    make_env_func=make_env_func)
        self.policy = Policy(config=config)
        self.policy_param = init_policy_param
        self.vector_env_size = config.actor.vector_env_size

        # init experience collector
        assert len(init_policy_param.gamma) == self.vector_env_size
        self.collectors = [ExperienceCollector(gamma=gamma, config=config) for gamma in init_policy_param.gamma]

        # init explorer
        self.explorers = [
            Explorer(action_size=config.model.action_size,
                     init_eps=self.policy_param.epsilon[i],
                     init_beta=config.intrinsic_reward.reward_ratio,
                     use_intrinsic_reward=config.intrinsic_reward.enable,
                     use_ucb=config.intrinsic_reward.use_ucb,
                     apply_value_scaling=config.apply_value_scaling) for i in range(self.vector_env_size)
        ]
        self.use_intrinsic_reward = config.intrinsic_reward.enable

    def step(self) -> List[SampleFromActor]:
        """Take an environment step.

        Returns:
            samples: List of sampled experiences.
        """
        if self.obs is None:
            # first reset
            self.obs = [env.reset() for env in self.envs]
            [c.on_reset(obs) for c, obs in zip(self.collectors, self.obs)]

        # take step
        prediction, intrinsic_reward = self.policy.infer(self.obs)
        q_e, q_i = prediction.as_numpy_tuple()

        # partially reset
        samples = []

        for i in range(self.vector_env_size):
            action = self.explorers[i].select_action(q_e[i], q_i[i] if self.use_intrinsic_reward else None)
            obs, reward, done, info = self.envs[i].step(action)
            sample = self.collectors[i].on_step(obs, reward, done, info, action, q_e[i],
                                                intrinsic_reward[i] if self.use_intrinsic_reward else None)
            if sample is not None: samples.append(sample)

            self.explorers[i].on_step(reward)
            self.obs[i] = obs

            # status
            self.reward_sum[i] += reward
            self.episode_len[i] += 1
            self.step_counter[i].step()
            if self.use_intrinsic_reward: self.intrinsic_reward_sum[i] += intrinsic_reward[i]

            # partially reset
            if done:
                obs = self.envs[i].reset()
                self.collectors[i].on_reset(obs)
                self.policy.on_partial_reset(i)
                self.explorers[i].on_done()
                self.obs[i] = obs

                # status
                self.reward_sum_mean[i].step(self.reward_sum[i])
                self.reward_sum[i] = 0.0
                self.episode_len_mean[i].step(self.episode_len[i])
                self.episode_len[i] = 0
                self.episode_counter[i].step()
                if self.use_intrinsic_reward:
                    self.intrinsic_reward_sum_mean[i].step(self.intrinsic_reward_sum[i])
                    self.intrinsic_reward_sum[i] = 0.0

        return samples

    def update_policy_param(self, policy_param: PolicyParam) -> None:
        """Update policy parameter.

        Args:
            policy_param: Parameter to be updated.
        """
        self.policy_param = policy_param
        [e.update_epsilon(eps) for e, eps in zip(self.explorers, self.policy_param.epsilon)]
        [c.update_gamma(g) for c, g in zip(self.collectors, self.policy_param.gamma)]

    def get_policy_param(self) -> PolicyParam:
        """Returns policy param."""
        return self.policy_param

    def update_model_param(self, param: bytes) -> None:
        """Update model parameter.

        Args:
            param: Byte expression of model parameter.
        """
        self.policy.update_model_param(param, only_online_model=False)


class ExperienceCollector:
    """Sampling batches from experiences.

    Args:
        gamma: Discount factor.
        config: CNN configuration.
    """
    def __init__(self, gamma: float, config: CNNConfigBase):
        self.sample_from_actor_def = config.sample_from_actor_def
        self.use_intrinsic_reward = config.intrinsic_reward.enable
        self.n_step = config.n_step
        self.gamma = gamma
        self.apply_value_scaling = config.apply_value_scaling
        self._clear_buffer()

    def update_gamma(self, gamma: float) -> None:
        """Update discount factor.

        Args:
            gamma: New parameter.
        """
        self.gamma = gamma

    def _clear_buffer(self) -> None:
        self.obs = deque([], maxlen=self.n_step + 1)
        self.reward = deque([], maxlen=self.n_step)
        self.action = deque([], maxlen=self.n_step)
        self.q_value = deque([], maxlen=self.n_step)
        if self.use_intrinsic_reward: self.intrinsic_reward = deque([], maxlen=self.n_step)

    def on_reset(self, obs: dict) -> None:
        """Called at on reset.

        Args:
            obs: Observation NumPy array.
        """
        self._clear_buffer()
        self.obs.append(obs)

    def on_step(self, obs: dict, reward: float, done: bool, info: dict, action: int, q_value: np.ndarray,
                intrinsic_reward: Optional[float]) -> Optional[SampleFromActor]:
        """Collect data at step.

        Args:
            obs: Observation NumPy array.
            reward: Reward float value.
            done: Boolean to tell episode is done or not.
            info: Info dictionary of episode.
            action: Action index.
            q_value: Q-value NumPy array.
            intrinsic_reward: Intrinsic reward value.
        """
        self.obs.append(obs)
        self.reward.append(reward)
        self.action.append(action)
        self.q_value.append(q_value)
        if self.use_intrinsic_reward: self.intrinsic_reward.append(intrinsic_reward)

        sample = None
        if len(self.obs) == self.obs.maxlen or done:
            sample = SampleFromActor.as_zeros(np_defs=self.sample_from_actor_def, size=1)
            discounted_reward_sum = sum([r * self.gamma**k for k, r in enumerate(self.reward)])

            if self.apply_value_scaling:
                target_q_value = np_value_scaling(discounted_reward_sum + self.gamma**self.n_step *
                                                  np_inverse_value_scaling(np.max(self.q_value[-1])) * (1 - done))
            else:
                target_q_value = discounted_reward_sum + self.gamma**self.n_step * np.max(self.q_value[-1]) * (1 - done)

            td_error = np.absolute(target_q_value - self.q_value[0][self.action[0]])

            # store data to sample
            sample.loss[0] = td_error
            sample.obs[0] = np.array(self.obs[0])
            sample.obs_next[0] = np.array(self.obs[-1])
            sample.action[0] = np.array(self.action[0])
            sample.discounted_reward_sum[0] = np.array(discounted_reward_sum)
            sample.gamma[0] = np.array(self.gamma)
            sample.is_done[0] = done
            if self.use_intrinsic_reward:
                discounted_intrinsic_reward_sum = sum([r * self.gamma**k for k, r in enumerate(self.intrinsic_reward)])
                sample.discounted_intrinsic_reward_sum[0] = np.array(discounted_intrinsic_reward_sum)

        return sample
