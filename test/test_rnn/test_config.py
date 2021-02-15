import tempfile
from pathlib import Path

from dqn.rnn.config import RNNConfigBase


def test_config():
    with tempfile.TemporaryDirectory() as log_dir:
        RNNConfigBase().save_as_yaml(Path(log_dir) / 'config.yaml')
