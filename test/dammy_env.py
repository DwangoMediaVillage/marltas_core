from typing import Tuple

import numpy as np


class DammyEnv:
    def __init__(self, max_step: int = 50):
        self.max_step = max_step
        self.t = 0

    def get_obs(self):
        return (np.random.random((4, 84, 84)) * 255).astype(dtype=np.uint8)

    def reset(self) -> np.ndarray:
        self.t = 0
        return self.get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.t += 1
        return self.get_obs(), np.random.random(), self.t >= self.max_step, {'score': 1}

    def close(self) -> None:
        pass
