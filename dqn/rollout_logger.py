"""Logger for episode rollouts."""
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from gym.wrappers.monitoring.video_recorder import ImageEncoder


class EpisodeDataSaver:
    """Helper class to save info of episode as json file."""
    def __init__(self):
        self.clear()

    def clear(self) -> None:
        """Clear stocked info."""
        self.action = []
        self.q_e = []
        self.q_i = []
        self.reward = []
        self.intrinsic_reward = []

    def save(self, filename: Path) -> None:
        """Saves info as json file.

        Args:
            filename: Path of saved json file.
        """
        with open(filename, 'w') as f:
            json.dump(
                {
                    'action': self.action,
                    'extrinsic_q_value': self.q_e,
                    'intrinsic_q_value': self.q_i,
                    'intrinsic_reward': self.intrinsic_reward,
                    'reward': self.reward,
                }, f)


class RolloutLogger:
    """Logs rollout data into disk.

    Args:
        out_dir: Path to put generated files.
        render: If true, `RolloutLogger` renders image states as movies.
        filename_header: Header of saved file's name.
    """
    def __init__(self,
                 out_dir: Path,
                 render: bool,
                 filename_header: str,
                 logger: logging.Logger = logging.getLogger(__name__)):
        assert out_dir.exists()
        self.out_dir = out_dir
        self.render = render
        self.filename_header = filename_header
        self.logger = logger
        self.n_episode = 0

        self.episode_data_saver = EpisodeDataSaver()
        self.encoder: Optional[ImageEncoder] = None

    def on_reset(self, obs: np.ndarray, image_frame: Optional[np.ndarray]) -> None:
        """Init image encoder at environment's reset.

        Args:
            obs: Observation.
            image_frame: List of NumPy array image frames.
        """
        if self.render:
            assert image_frame is not None
            self.encoder = ImageEncoder(output_path=str(self.out_dir / f'{self.filename_header}_{self.n_episode}.mp4'),
                                        frame_shape=image_frame.shape,
                                        frames_per_sec=30,
                                        output_frames_per_sec=30)
            self.encoder.capture_frame(image_frame)

    def on_step(self, action: int, q_e: List[float], q_i: Optional[List[float]], intrinsic_reward: Optional[float],
                obs: np.ndarray, reward: float, info: dict, done: bool, image_frame: Optional[np.ndarray]) -> None:
        """On taking a step of environment.

        Args:
            action: Action index policy took
            q_e: Extrinsic q-value inferred by policy
            q_i: Intrinsic q-value inferred by policy
            intrinsic_reward: Intrinsic reward
            obs: observation
            reward: Extrinsic reward
            info: Info dictionary given by environment step
            done: Episode terminates or not.
            image_frame: List of image frames for movie rendering.
        """
        self.episode_data_saver.action.append(int(action))
        self.episode_data_saver.q_e.append(q_e)
        self.episode_data_saver.q_i.append(q_i)
        self.episode_data_saver.intrinsic_reward.append(intrinsic_reward)
        self.episode_data_saver.reward.append(float(reward))

        if self.render and image_frame is not None:
            self.encoder.capture_frame(image_frame)

        if done:
            # end of episode
            self.logger.info(f'[{self.filename_header}] End of episode {self.n_episode}')
            self.episode_data_saver.save(self.out_dir / f'{self.filename_header}_{self.n_episode}.json')
            if self.render:
                assert self.encoder is not None
                self.encoder.close()
                self.encoder = None
            self.n_episode += 1
