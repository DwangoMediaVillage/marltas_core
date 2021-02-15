from test.dammy_env import DammyEnv

import numpy as np

from dqn.policy import PolicyParam
from dqn.rnn.actor import Actor, ExperienceCollector
from dqn.rnn.config import ActorConfig, ModelConfig, RNNConfigBase
from dqn.rnn.model import ModelState


def test_experience_collector(config=RNNConfigBase()):
    # dammy data
    state = ModelState.zeros(size=config.model.lstm_hidden_size, use_intrinsic_model=config.intrinsic_reward.enable)
    obs = np.zeros(config.obs_shape)
    q_value = np.random.random(config.model.action_size)
    action = 4
    config = RNNConfigBase(model=ModelConfig(action_size=action))

    # init collector
    collector = ExperienceCollector(gamma=0.99, config=config)

    # on_step returns None until SEQ_LEN steps
    collector.on_reset(obs, state)
    for _ in range(config.seq_len - 1):
        assert collector.on_step(obs, 0.1, False, {}, action, q_value, state, 0.1) is None

    # returns sample
    sample = collector.on_step(obs, 0.1, False, {}, action, q_value, state, 0.1)
    assert sample is not None
    sample.validate_type(np_defs=config.sample_from_actor_def)

    # window skip
    for _ in range(config.actor.window_skip - 1):
        assert collector.on_step(obs, 0.1, False, {}, action, q_value, state, 0.1) is None
    sample = collector.on_step(obs, 0.1, False, {}, action, q_value, state, 0.1)
    assert sample is not None
    sample.validate_type(np_defs=config.sample_from_actor_def)


def test_experience_collector_early_done(config=RNNConfigBase()):
    # dammy data
    state = ModelState.zeros(size=config.model.lstm_hidden_size, use_intrinsic_model=config.intrinsic_reward.enable)
    obs = np.zeros(config.obs_shape)
    q_value = np.random.random(config.model.action_size)
    action = 4
    config = RNNConfigBase(model=ModelConfig(action_size=action))

    # init collector
    collector = ExperienceCollector(gamma=0.99, config=config)
    collector.on_reset(obs, state)

    # done at the first step
    sample = collector.on_step(obs, 0.1, True, {}, action, q_value, state, 0.1)
    assert sample is not None
    sample.validate_type(np_defs=config.sample_from_actor_def)
    assert all(sample.is_done[0])

    # init collector again
    collector = ExperienceCollector(gamma=0.99, config=config)
    collector.on_reset(obs, state)
    for _ in range(config.seq_len - 3):
        collector.on_step(obs, 0.1, False, {}, action, q_value, state, 0.1)
    sample = collector.on_step(obs, 0.1, True, {}, action, q_value, state, 0.1)
    assert all(sample.is_done[0, config.seq_len - 3:])


def test_actor(config=RNNConfigBase()):
    actor = Actor(init_policy_param=PolicyParam(epsilon=np.random.random(2), gamma=np.ones(2) * 0.99),
                  config=RNNConfigBase(actor=ActorConfig(vector_env_size=2)),
                  make_env_func=lambda c: DammyEnv(max_step=40))

    # can update policy param
    actor.update_policy_param(PolicyParam(epsilon=np.random.random(2), gamma=np.ones(2) * 0.99))

    # step
    for _ in range(40):
        samples = actor.step()
        if len(samples): [s.validate_type(np_defs=config.sample_from_actor_def) for s in samples]
