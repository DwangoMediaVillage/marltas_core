import tempfile
from pathlib import Path

from dqn.cnn.config import CNNConfigBase, IntrinsicRewardConfig, LearnerConfig
from dqn.cnn.datum import Batch, SampleFromBuffer
from dqn.cnn.learner import Learner
from dqn.episodic_curiosity import EpisodicCuriosityConfig


def test_vector_obs_update():
    config = CNNConfigBase(obs_shape=[
        2,
    ],
                           intrinsic_reward=IntrinsicRewardConfig(
                               enable=True, episodic_curiosity=EpisodicCuriosityConfig(enable=True)))
    learner = Learner(config=config)
    batch = Batch.from_buffer_sample(sample=SampleFromBuffer.as_random(size=3, np_defs=config.sample_from_buffer_def))
    learner.update_core(batch)


def test_learner():
    config = CNNConfigBase(learner=LearnerConfig(batch_size=3, gpu_id=None, target_sync_interval=1, double_dqn=True),
                           intrinsic_reward=IntrinsicRewardConfig(
                               enable=True, episodic_curiosity=EpisodicCuriosityConfig(enable=True)))
    learner = Learner(config=config)

    loss = learner.update(
        Batch.from_buffer_sample(sample=SampleFromBuffer.as_random(size=3, np_defs=config.sample_from_buffer_def)))
    loss.validate_type(np_defs=config.loss_def)
    assert len(loss) == 3

    stat = learner.get_status()
    assert stat.online_update == 1
    assert stat.target_update == 1
    assert isinstance(stat.extrinsic_loss, float)
    assert isinstance(stat.extrinsic_td_error_mean, float)
    assert isinstance(stat.extrinsic_q_value_mean, float)
    assert isinstance(stat.intrinsic_loss, float)
    assert isinstance(stat.intrinsic_td_error_mean, float)
    assert isinstance(stat.intrinsic_q_value_mean, float)

    learner.get_model_param()

    learner.double_dqn = True
    learner.update(
        Batch.from_buffer_sample(sample=SampleFromBuffer.as_random(size=3, np_defs=config.sample_from_buffer_def)))


def test_save_load_model():
    learner = Learner(config=CNNConfigBase(intrinsic_reward=IntrinsicRewardConfig(enable=False),
                                           learner=LearnerConfig(gpu_id=None, batch_size=1, target_sync_interval=1)))
    learner_load = Learner(
        config=CNNConfigBase(intrinsic_reward=IntrinsicRewardConfig(enable=False),
                             learner=LearnerConfig(gpu_id=None, batch_size=1, target_sync_interval=1)))
    assert learner.get_model_param() != learner_load.get_model_param()

    with tempfile.TemporaryDirectory() as log_dir:
        log_dir = Path(log_dir)
        learner.save_model(log_dir=log_dir, global_step=0)
        snap_filename = log_dir / f'online_model_0.pkl'
        learner_load.load_online_model(snap_filename)
        assert learner.get_model_param() == learner_load.get_model_param()
