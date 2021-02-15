from test.dammy_env import DammyEnv

import numpy as np
import torch

from dqn.episodic_curiosity import (EpisodicCuriosityConfig,
                                    EpisodicCuriosityModule)


def test_compute_reward():
    module = EpisodicCuriosityModule(config=EpisodicCuriosityConfig(enable=True),
                                     obs_shape=[4, 84, 84],
                                     action_size=6,
                                     vector_env_size=1)

    env = DammyEnv()
    obs = env.reset()

    for t in range(module.config.k + 2):
        x = torch.from_numpy(np.stack([obs.astype(np.float32)]))
        r = module(x)
        assert not np.isnan(r) and not np.isinf(r) and r is not None
        if t + 1 < module.config.k:
            assert r == 0

        obs, _, _, _ = env.step(0)
    assert len(module.memory[0]) > 0
    assert module.distance_average[0] != np.nan
    module.partial_reset(0)
    assert len(module.memory[0]) == 0
