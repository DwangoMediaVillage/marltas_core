"""RNN-DQN actor."""
from collections import deque
from typing import Callable, List, Optional, Union

import gym
import numpy as np

from dqn.actor import ActorBase
from dqn.explorer import Explorer
from dqn.make_env import EnvConfig, make_env
from dqn.policy import PolicyParam
from dqn.rnn.config import RNNConfigBase
from dqn.rnn.datum import SampleFromActor
from dqn.rnn.model import ModelState
from dqn.rnn.policy import Policy
from dqn.utils import np_inverse_value_scaling, np_value_scaling


class Actor(ActorBase):
    """Actor for sampling sequential samples from vectorized envs.

    Args:
        init_policy_param: Initial policy parameter.
        config: RNN configuration.
        process_index: Index of process.
        make_env_func: Function to init environment.
    """
    def __init__(self,
                 init_policy_param: PolicyParam,
                 config: RNNConfigBase,
                 process_index: int = 0,
                 make_env_func: Callable[[EnvConfig], gym.Env] = make_env):
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

        # initial model hidden state
        self.policy_state: List[ModelState] = [self.policy.model.get_init_state() for _ in range(self.vector_env_size)]

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
        """Take environment step.

        Returns:
            samples: List of sampled experiences.
        """
        if self.obs is None:
            # first reset
            self.obs = [env.reset() for env in self.envs]
            [c.on_reset(obs, state) for c, obs, state in zip(self.collectors, self.obs, self.policy_state)]

        # take step
        prediction, intrinsic_reward, self.policy_state = self.policy.infer(self.obs, self.policy_state)
        q_e, q_i = prediction.as_numpy_tuple()

        # partially reset
        samples = []
        for i in range(self.vector_env_size):
            action = self.explorers[i].select_action(q_e[i], q_i[i] if self.use_intrinsic_reward else None)
            obs, reward, done, info = self.envs[i].step(action)
            sample = self.collectors[i].on_step(
                obs=obs,
                reward=reward,
                done=done,
                info=info,
                action=action,
                q_value=q_e[i],
                state=self.policy_state[i],
                intrinsic_reward=intrinsic_reward[i] if self.use_intrinsic_reward else None)
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
                state = self.policy.model.get_init_state()
                self.policy.on_partial_reset(i)
                self.collectors[i].on_reset(obs, state)
                self.explorers[i].on_done()

                self.obs[i] = obs
                self.policy_state[i] = state

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
    """Samping sequential samples from experiences.

    Args:
        gamma: Discount factor.
        config: RNN configuration.
    """
    def __init__(self, gamma: float, config: RNNConfigBase):
        self.sample_from_actor_def = config.sample_from_actor_def
        self.seq_len = config.seq_len
        self.gamma = gamma
        self.use_intrinsic_reward = config.intrinsic_reward.enable
        self.n_step = config.n_step
        self.window_skip = config.actor.window_skip
        self.apply_value_scaling = config.apply_value_scaling

        self.discount = np.geomspace(1, self.gamma**self.n_step, num=self.n_step, endpoint=False)

        eta = 0.9
        self.gather_td_error = lambda td_error: eta * np.max(td_error) + (1 - eta) * np.mean(td_error)

        self._clear_buffer()

    def update_gamma(self, gamma: float) -> None:
        """Update discount factor.

        Args:
            gamma: New parameter.
        """
        self.gamma = gamma

    def _clear_buffer(self) -> None:
        self.obs = deque([], maxlen=self.seq_len + 1)
        self.reward = deque([], maxlen=self.seq_len)
        if self.use_intrinsic_reward: self.intrinsic_reward = deque([], maxlen=self.seq_len)
        self.action = deque([], maxlen=self.seq_len)
        self.q_value = deque([], maxlen=self.seq_len)
        self.state = deque([], maxlen=self.seq_len + 1)
        self.step_after_full = 0

    def on_reset(self, obs: Union[dict], state: ModelState) -> None:
        """Called at on reset.

        Args:
            obs: Observation NumPy array.
            state: Hidden states of RNN.
        """
        self._clear_buffer()
        self.obs.append(obs)
        self.state.append(state)

    def on_step(self, obs: Union[dict], reward: float, done: bool, info: dict, action: int, q_value: np.ndarray,
                state: ModelState, intrinsic_reward: Optional[float]) -> Optional[SampleFromActor]:
        """Collect data at step.

        Args:
            obs: Observation NumPy array.
            reward: Reward float value.
            done: Boolean to tell episode is done or not.
            info: Info dictionary of episode.
            action: Action index.
            prediction: Prediction of Q-Network(s).
            intrinsic_reward: Intrinsic reward value.
        """
        self.obs.append(obs)
        self.reward.append(reward)
        if self.use_intrinsic_reward: self.intrinsic_reward.append(intrinsic_reward)
        self.action.append(action)
        self.state.append(state)
        self.q_value.append(q_value)

        if done:
            return self.pack_sample(done)

        if len(self.reward) == self.seq_len:
            sample: Optional[SampleFromActor] = None
            if self.step_after_full % self.window_skip == 0: sample = self.pack_sample(done)
            self.step_after_full += 1
            return sample

        return None

    def pack_sample(self, is_episode_done: bool) -> SampleFromActor:
        """Pack experience data as `SampleFromActor`.
        Args:
            is_episode_done: End of episode or not.

        Returns:
            sample: Sample generated by actor.
        """
        sample = SampleFromActor.as_zeros(size=1, np_defs=self.sample_from_actor_def)

        L = len(self.reward)

        is_done = np.zeros(self.seq_len, dtype=np.bool)
        if is_episode_done: is_done[L - 1:] = True

        # compute initial priority as loss
        reward = np.array(self.reward)
        action = np.array(self.action)
        discounted_reward_sum = np.correlate(np.pad(reward, (0, self.discount.size - 1), 'constant'), self.discount)
        q_value = np.array(self.q_value)
        if self.apply_value_scaling:
            target_value = np_value_scaling(discounted_reward_sum + self.gamma**self.n_step *
                                            np_inverse_value_scaling(np.max(q_value, axis=1) * (1 - is_done[:L])))
        else:
            target_value = discounted_reward_sum + self.gamma**self.n_step * np.max(q_value, axis=1) * (1 - is_done[:L])
        prediction = np.choose(action, q_value.T)

        sample.loss[0] = self.gather_td_error(np.absolute(target_value - prediction))
        sample.obs[0, :(L + 1)] = np.array([np.array(o) for o in self.obs])
        sample.action[0, :L] = action
        sample.discounted_reward_sum[0, :L] = discounted_reward_sum
        sample.gamma[0] = self.gamma
        sample.is_done[0] = is_done

        h_e, c_e, h_i, c_i = self.state[0].as_numpy_tuple()
        sample.lstm_hidden_h_e[0] = h_e
        sample.lstm_hidden_c_e[0] = c_e

        if self.use_intrinsic_reward:
            sample.discounted_intrinsic_reward_sum[0, :L] = np.correlate(
                np.pad(np.array(self.intrinsic_reward), (0, self.discount.size - 1), 'constant'), self.discount)
            sample.lstm_hidden_h_i[0] = h_i
            sample.lstm_hidden_c_i[0] = c_i

        return sample
