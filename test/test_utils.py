import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dqn.utils import (ConfigBase, Counter, CustomMetrics, EventObject,
                       MovingAverage, init_log_dir, pad_along_axis)


def test_config_base():
    @dataclass
    class ConfigB(ConfigBase):
        int_data: int = 1
        float_data: float = 2.0
        str_data: str = '3'
        path_data: Path = Path('foo/bar')

    @dataclass
    class ConfigA(ConfigBase):
        int_data: int = 2
        float_data: float = 3.0
        str_data: str = '4'
        path_data: Path = Path('bar/baz')
        config_data: ConfigB = ConfigB()

    with tempfile.TemporaryDirectory() as d:
        yaml_path = Path(d) / 'config.yaml'

        # save as as yaml
        config = ConfigA()
        config.save_as_yaml(yaml_path)

        # can restore from yaml
        assert ConfigA.load_from_yaml(yaml_path) == config


def test_init_log_dir():
    @dataclass
    class Config(ConfigBase):
        foo: str = 'aaabbbccc'

    with tempfile.TemporaryDirectory() as dname:
        log_dir = Path(dname)
        init_log_dir(log_dir, Config())


def test_moving_average():
    mean = MovingAverage(0.9)
    assert mean.average is None
    mean.step(1)
    assert isinstance(mean.average, (int, float))
    mean.step(2)
    assert isinstance(mean.average, (int, float))


def test_counter():
    c = Counter()
    assert c.count == 0
    c.step(10)
    assert c.get_count_per_sec() is None
    c.step()
    assert c.get_count_per_sec() > 0
    assert c.count == 11


def test_pad_along_axis():
    x = np.ones((3, 4, 5))
    y = pad_along_axis(x, 1, (0, 3), 'constant')
    assert y.shape == (3, 7, 5)
    assert y[:, 4:, :].sum() == 0.0


def test_custom_metrics():
    metric_keys = ['foo', 'bar']
    metric_types = ['sum', 'last']

    m = CustomMetrics(metric_keys=metric_keys, metric_types=metric_types)
    m.take_from_info({'foo': 1, 'bar': 0.1, 'baz': 2.0})
    result = m.as_dict()
    assert 'foo' in result
    assert 'bar' in result

    pickle.dumps(m)  # can dump


def test_event_object():
    @dataclass
    class C(EventObject):
        int_data: int = 1
        float_data: float = 2.0
        numpy_data: np.ndarray = np.random.random(3)
        numpy_scalar_data: np.ndarray = np.array([4])
        list_data: List[float] = field(default_factory=lambda: [5.0, 6.0, 7.0])
        dict_data: dict = field(default_factory=lambda: {'foo': 8, 'bar': [9.0, 10.0]})

    with tempfile.TemporaryDirectory() as log_dir:
        with SummaryWriter(log_dir=Path(log_dir)) as writer:
            C().write_summary(writer, 1)
