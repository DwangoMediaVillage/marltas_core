import tempfile
from pathlib import Path

from dqn import episodic_curiosity
from dqn.learner import LearnerConfig
from dqn.rnn.config import IntrinsicRewardConfig, RNNConfigBase
from dqn.rnn.datum import Batch, SampleFromBuffer
from dqn.rnn.learner import Learner
from dqn.rnn.policy import Policy


def test_vector_obs_update():
    config = RNNConfigBase(obs_shape=[
        2,
    ],
                           intrinsic_reward=IntrinsicRewardConfig(
                               enable=True, episodic_curiosity=episodic_curiosity.EpisodicCuriosityConfig(enable=True)))

    learner = Learner(config=config)
    batch = Batch.from_buffer_sample(sample=SampleFromBuffer.as_random(size=3, np_defs=config.sample_from_buffer_def))
    learner.update_core(batch)


def test_update():
    config = RNNConfigBase(learner=LearnerConfig(batch_size=3, double_dqn=True, target_sync_interval=1),
                           intrinsic_reward=IntrinsicRewardConfig(
                               enable=True, episodic_curiosity=episodic_curiosity.EpisodicCuriosityConfig(enable=True)))
    learner = Learner(config=config)
    batch = Batch.from_buffer_sample(sample=SampleFromBuffer.as_random(size=3, np_defs=config.sample_from_buffer_def))
    learner.update_core(batch)


def test_save_model():
    config = RNNConfigBase(learner=LearnerConfig(batch_size=2, target_sync_interval=1),
                           intrinsic_reward=IntrinsicRewardConfig(enable=True))
    learner = Learner(config=config)

    with tempfile.TemporaryDirectory() as log_dir:
        learner.save_model(log_dir=Path(log_dir), global_step=0)


def test_update_param():
    config = RNNConfigBase(learner=LearnerConfig(batch_size=3, double_dqn=True, target_sync_interval=1),
                           intrinsic_reward=IntrinsicRewardConfig(
                               enable=True, episodic_curiosity=episodic_curiosity.EpisodicCuriosityConfig(enable=True)))
    learner = Learner(config=config)
    policy = Policy(config=config)

    assert learner.online_model.get_param() != policy.model.get_param()
    assert learner.episodic_curiosity_module.embedding_network.get_param(
    ) != policy.episodic_curiosity_module.embedding_network.get_param()
    assert learner.episodic_curiosity_module.inverse_model.get_param(
    ) != policy.episodic_curiosity_module.inverse_model.get_param()

    policy.update_model_param(learner.get_model_param(), only_online_model=False)
    assert learner.online_model.get_param() == policy.model.get_param()
    assert learner.episodic_curiosity_module.embedding_network.get_param(
    ) == policy.episodic_curiosity_module.embedding_network.get_param()
    assert learner.episodic_curiosity_module.inverse_model.get_param(
    ) == policy.episodic_curiosity_module.inverse_model.get_param()
