import tempfile
from pathlib import Path

import numpy as np

from dqn.rollout_logger import RolloutLogger


def test_rollout_logger():
    def gen_dammy_image_frame():
        return (np.random.random((128, 128, 3)) * 255).astype(np.uint8)

    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        logger = RolloutLogger(out_dir=out_dir, render=True, filename_header='')

        logger.on_reset(obs=None, image_frame=gen_dammy_image_frame())
        for i in range(10):
            logger.on_step(action=0,
                           q_e=0.123,
                           q_i=0.2525,
                           intrinsic_reward=0.11,
                           obs=None,
                           reward=0.222,
                           info={'score': 10},
                           done=i == 9,
                           image_frame=gen_dammy_image_frame())
