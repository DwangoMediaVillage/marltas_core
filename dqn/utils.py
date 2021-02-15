"""Util function/class implementations."""
import inspect
import pickle
import pprint
import random
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

T = TypeVar('T', bound='Parent')


@dataclass
class ConfigBase:
    """Serializable config as yaml"""
    def save_as_yaml(self, yaml_path: Path, ignored_top_keys: Optional[List[str]] = None) -> None:
        """Export config as yaml file.

        Args:
            yaml_path: Filename of output file.
            ignored_top_keys: Keys to be ignored.
        """
        def encode_config(dict_config):
            for key, val in dict_config.items():
                if isinstance(val, Path):
                    dict_config[key] = str(val)
                elif isinstance(val, dict):
                    dict_config[key] = encode_config(val)
                else:
                    pass
            return dict_config

        with open(yaml_path, 'w') as f:
            config_dict = asdict(self)
            if ignored_top_keys is not None:
                config_dict = {k: v for k, v in config_dict.items() if not k in ignored_top_keys}
            yaml.dump(encode_config(config_dict), f)

    @classmethod
    def load_from_yaml(cls: Type[T], yaml_path: Path, allow_illegal_keys=False) -> T:
        """Load config from yaml file.
        Args:
            yaml_path: Filename of yaml file.
            allow_illegal_keys: If true, illegal key will not cause exception.
        """
        assert yaml_path.exists()

        def decode_config(parent_cls, dict_config, config_filter):
            if dict_config is None: return dict_config
            for key, val in dict_config.items():
                if key in parent_cls.__dataclass_fields__:
                    config_filter[key] = True
                    child_cls = parent_cls.__dataclass_fields__[key].type
                    if child_cls == Path:
                        dict_config[key] = Path(val)
                    elif inspect.isclass(child_cls) and issubclass(child_cls, ConfigBase):
                        child_val, config_filter = decode_config(child_cls, val, config_filter)
                        dict_config[key] = child_cls(**child_val)
                    else:
                        pass
                else:
                    config_filter[key] = False
            return dict_config, config_filter

        def filter_config(decoded, config_filter):
            for key, val in config_filter.items():
                if isinstance(val, bool):
                    if not val: decoded.pop(key)
                else:
                    filter_config(decoded[key], val)
            return decoded

        with open(yaml_path) as f:
            decoded, config_filter = decode_config(cls, yaml.full_load(f), {})
            if allow_illegal_keys:
                decoded = filter_config(decoded, config_filter)  # remove values whose name is not in members.
            if decoded is None:
                return cls()
            else:
                return cls(**decoded)


class EventObject:
    """Serializable dataclass.

    Note that the all fields should be pickalable.
    """
    @classmethod
    def from_bytes(cls: Type[T], bytes_data: bytes) -> T:
        return cls(**pickle.loads(bytes_data))

    def to_bytes(self) -> bytes:
        return pickle.dumps(asdict(self))

    def __eq__(self, other: T) -> bool:
        check = []
        for k, v in asdict(self).items():
            other_v = getattr(other, k)
            if isinstance(v, np.ndarray):
                check.append(np.array_equal(v, other_v))
            else:
                check.append(v == other_v)
        return all(check)

    def _do_write_summary(self, value: Any, writer: SummaryWriter, global_step: int, namespace: Optional[Path]) -> None:
        if isinstance(value, (int, float)):
            writer.add_scalar(str(namespace), value, global_step)
        elif isinstance(value, (np.ndarray, np.generic)):
            if value.ndim == 1 and len(value) == 1: writer.add_scalar(str(namespace), value, global_step)
            writer.add_histogram(str(namespace), value, global_step)
        elif isinstance(value, (list, tuple)):
            writer.add_histogram(str(namespace), np.array(value), global_step)
        elif isinstance(value, dict):
            [self._do_write_summary(v, writer, global_step, namespace / k) for k, v in value.items()]
        else:
            raise NotImplementedError(f"Invalid value type: {value}")

    def write_summary(self, writer: SummaryWriter, global_step: int, namespace: Optional[Path] = None) -> None:
        """Write status to tfevent"""
        if namespace is None: namespace = Path(type(self).__name__)
        for name, v in asdict(self).items():
            if v is None: continue  # skip none value
            self._do_write_summary(v, writer, global_step, namespace / name)

    def as_format_str(self) -> str:
        return pprint.pformat(asdict(self), compact=True)


def init_log_dir(log_dir: Path, config: ConfigBase) -> None:
    """Initialize logging directory.

    Args:
        log_dir: Directory to put log data.
        config: Configuration object to be saved.
    """
    # dump config as yaml
    config.save_as_yaml(log_dir / 'config.yaml')

    # save git status
    def save_output(save_path, command):
        with open(save_path, 'wb') as f:
            f.write(subprocess.check_output(command.split()))

    save_output(log_dir / "git-head.txt", "git rev-parse HEAD")
    save_output(log_dir / "git-status.txt", "git status")
    save_output(log_dir / "git-log.txt", "git log")
    save_output(log_dir / "git-diff.txt", "git diff")


class EpsilonSampler:
    """Stratified sampling of epsilon-greedy parmeters based on p.6 of [Ape-X paper](http://arxiv.org/abs/1803.00933).

    Args:
        init_eps_base: Initial `epsilon-base`.
        final_eps_base: Final `epsilon-base`.
        init_alpha: Initial `alpha`.
        final_alpha: Final `alpha`.
        final_step: Final step.
        min_eps: Minimum value of sampled epsilons.
    """
    def __init__(self, init_eps_base: float, final_eps_base: float, init_alpha: float, final_alpha: float,
                 final_step: int, min_eps: float):
        self.f_eps = lambda s: init_eps_base + (final_eps_base - init_eps_base) * (min(s, final_step) / final_step)
        self.f_alpha = lambda s: init_alpha + (final_alpha - init_alpha) * (min(s, final_step) / final_step)
        self.eps_base = init_eps_base
        self.alpha = init_alpha
        self.min_eps = min_eps

    def __call__(self, step: int, index: np.ndarray, max_index: int) -> np.ndarray:
        """Returns sampled epsilon values as numpy array.

        Args:
            index: Indices of vectorized actors.
            max_index: Total size of actors.
        returns:
            epsilons: Array of epsilon values.
        """
        self.eps_base = self.f_eps(step)
        self.alpha = self.f_alpha(step)
        x = np.clip(index + np.random.normal(0.0, scale=0.1, size=len(index)), 0, index.max())
        return np.maximum(np.power(self.eps_base, 1 + (x / max_index) * self.alpha), self.min_eps)


class MovingAverage:
    """Helper class to compute moving average.

    Args:
        decay: Moving decay.
        initial_value: Initial value.
    """
    def __init__(
        self,
        decay: float,
        initial_value: Optional[Union[float, np.ndarray]] = None,
    ):
        self.average_value = initial_value
        self.decay = decay

    def step(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Update average.

        Args:
            value: Value to be accumulated.
        """
        if self.average_value is None:
            self.average_value = value
        else:
            self.average_value += (1 - self.decay) * (value - self.average_value)
        return self.average_value

    @property
    def average(self) -> Optional[Union[float, np.ndarray]]:
        return self.average_value


class Counter:
    """Helper class to count values.

    Args:
        buffer_len: Size of counting buffer.
    """
    def __init__(self, buffer_len=10):
        assert buffer_len >= 2
        self.walltime = deque([], maxlen=buffer_len)
        self.count_history = deque([], maxlen=buffer_len)
        self.t = 0

    def step(self, step_size: int = 1) -> None:
        """Count up.

        Args:
            step_size: Size of step.
        """
        self.t += step_size
        self.count_history.append(self.t)
        self.walltime.append(time.time())

    @property
    def count(self) -> int:
        return self.t

    def get_count_per_sec(self) -> Optional[float]:
        """Returns number of steps per seconds."""
        if len(self.walltime) >= 2:
            time_diff = self.walltime[-1] - self.walltime[0]
            count_diff = self.count_history[-1] - self.count_history[0]
            return count_diff / time_diff
        else:
            return None


def none_mean(array: Optional[List[Optional[Union[float, int]]]]) -> Optional[float]:
    """Returns mean of number array with None.

    Args:
        array: Array-like object which may inclues `None`.

    Returns:
        mean: Mean of `array`, or `None` if `array` hes not have any non none values.
    """
    if array is None: return None
    if len(array) == 0: return None
    # remove None
    array = [a for a in array if a is not None]
    return np.mean(array) if len(array) else None


def pad_along_axis(array: np.ndarray, axis: int, pad_width: Tuple[int, int], mode: str) -> np.ndarray:
    """Pad along arbitrary axis.

    Args:
        array: NumPy array input.
        axis: Axis to pad.
        pad_width: Width of padding.
        mode: One of the following string values or a user supplied function.
    """
    pad = [(0, 0)] * array.ndim
    pad[axis] = pad_width
    return np.pad(array, pad, mode=mode)


class CustomMetrics:
    """Custom metric for evaluation.

    Args:
        metric_keys: Key of metric. These keys are utilized to extract metric value from OpenAI gym environment's `info` dict.
        metric_types: Data type of metric `sum`, `average`, and `last` are supported.
    """
    def __init__(self, metric_keys: List[str], metric_types: List[str]):
        self.metrics_keys = metric_keys
        self.metrics_types = metric_types
        self.step = 0
        self.data = {}

    def take_from_info(self, info: dict) -> None:
        """Extract metric data from `info' dict.

        Args:
            info: Dictionary given by OpenAI gym environment.
        """
        self.step += 1
        for k, t in zip(self.metrics_keys, self.metrics_types):
            if not k in info: continue
            if t == 'sum' or t == 'average':
                self.data[k] = self.data.get(k, 0.0) + info[k]
            elif t == 'last':
                self.data[k] = info[k]
            else:
                raise NotImplementedError

    def as_dict(self) -> dict:
        """Returns dictionary expression of the extracted metrics.

        Returns:
            metrics: Metrics as an dictionary.
        """
        res = {}
        for k, t in zip(self.metrics_keys, self.metrics_types):
            if not k in self.data: continue
            if t == 'average':
                if self.step > 0: res[k] = self.data[k] / self.step
            elif t == 'last' or t == 'sum':
                res[k] = self.data[k]
            else:
                raise NotImplementedError
        return res


def init_random_seed(seed: int) -> None:
    """Initialize random seed of Python's random module, NumPy, and PyTorch.
    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def np_value_scaling(z: np.ndarray, eps: float = 0.001) -> np.ndarray:
    """Scale NumPy Q-value.

    Args:
        z: Input Q-value.
    """
    return np.sign(z) * (np.sqrt(np.absolute(z) + 1) - 1) + eps * z


def np_inverse_value_scaling(z: np.ndarray, eps: float = 0.001) -> np.ndarray:
    """Inverse function of `np_value_scaling`."""
    n = np.sqrt(1 + 4 * eps * (np.absolute(z) + 1 + eps)) - 1
    return np.sign(z) * (n / (2 * eps) - 1)


def value_scaling(z: torch.Tensor, eps=0.001) -> torch.Tensor:
    """Scale PyTorch Q-value.

    Args:
        z: Input Q-value.
    """
    return torch.sign(z) * (torch.sqrt(torch.abs(z) + 1) - 1) + eps * z


def inverse_value_scaling(z: torch.Tensor, eps=0.001) -> torch.Tensor:
    """Inverse function of `value_scaling`."""
    n = torch.sqrt(1 + 4 * eps * (torch.abs(z) + 1 + eps)) - 1  # numerator
    return torch.sign(z) * ((n / (2 * eps)) - 1)
