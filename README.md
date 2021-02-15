# MarltasCore

A PyTorch-based distributed asynchronized Deep Q-learning. The training modules are distributed to different processes and nodes. Communication between the modules is implemented by [gRPC](https://grpc.io/). The concept of architecture design is available on [dmv.nico](https://dmv.nico/en/articles/marltas_core/) page.

## Supported DQN algorithms

- [Recurrent neural network as Q-Network (R2D2)](https://openreview.net/pdf?id=r1lyTjAqYX)
    - Only `stored state` style is supported.
- [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)
- [Double Q-learning](http://arxiv.org/abs/1509.06461)
- [Dueling Network](http://arxiv.org/abs/1511.06581)
- [Intrinsic reward using Random Network Distillation and episodic curiosity (NGU)](http://arxiv.org/abs/2002.06038)
- [Epsilon-greedy parameter using upper confidence bound](http://arxiv.org/abs/2003.13350)
    - Note that we use Sliding-window UCB-1 Tuned to implement meta-controller, and discount factor exploration is not supported.


## How to run experiments

### Init python environment

Install the following libraries in addition to [PyTorch](https://pytorch.org/).

```bash
pip install atari-py grpcio-tools "gym[atari]" pyyaml tensorboard
```

### Build gRPC codes

Generate Python codes for gRPC communication by GNU Make.

```bash
make
```

### Run experiment

There are the four python scripts in `bin` directory.

|                                             | RNN-DQN              | CNN-DQN              |
|---------------------------------------------|----------------------|----------------------|
| Synchronized training (only on single node) | `sync_rnn_train.py`  | `sync_cnn_train.py`  |
| Asynchronized training                      | `async_rnn_train.py` | `async_cnn_train.py` |

Execute one of them to run experiments:

```bash
PYTHONPATH=. python bin/async_rnn_train.py
```

`-h` option shows the help doc of arguments.

### Configuration guide
　
You can edit the hyperparameters and switch the training methods by YAML file. The structure of the configuration follows the structure of `dataclass` objects in `dqn/rnn/config.py` or `dqn/cnn/config.py`. Meanings of the config tags are also available on docstring of each config class.

```yaml
### Example of async-rnn train's configuration ###

# URL and port of gRPC server. In case of multi-node training, `localhost` must be changed to hostname of the head node.
actor_manager_url: localhost:1111
evaluator_url: localhost:2222
param_distributor_url: localhost:3333
replay_buffer_url: localhost:4444

seq_len: 16  # Sequential length of training sample
n_actor_process: 3  # Number of actor(s) in a node
n_step: 3  # N of n-step learning algorithm
gamma: 0.997  # Discount factor

actor:
  vector_env_size: 3  # vector env size at a actor process
  window_skip: 8  # Time step interval to collect sequence sample

env:
  name: PongNoFrameskip-v4  # name of gym.make
  wrap_atari: true  # If true, env is wrapped with OpenAI baseline style

evaluator:
  n_eval: 3  # Number of episodes for each evaluation

learner:
  adam_lr: 0.0001  # Learning rate of Adam optimizer
  batch_size: 32  # Batch size
  gpu_id: 0  # GPU id for GPGPU
  target_sync_interval: 1000  # Interval of target network update

model:
  action_size: 6  # Dimensional size of action

replay_buffer:
  capacity: 10000  # Maximum size of replay buffer

trainer:  # Intervals are in the number of online network updates
  eval_interval: 10000  # Evaluation interval
  initial_sample_size: 100  # Learner waits until the size of replay buffer reaches this size
  log_interval: 500  # Interval to take log and save as tfevent file
  max_global_step: 150000  # Maximum size of update
  param_sync_interval: 1  # Interval to send Learner's parameter to actors
  snap_interval: 150000  # Interval to take snapshots of DQN model(s)
```

### Visualize training log

Training log will be written to `tfevent` file which can be visualized by `tensorboard`.

### Enjoy with a trained DQN-agent

Run `bin/rollout_rnn.py` or `bin/rollout_cnn.py`, with snapshot files generated by the training scripts.

### Experiment with your own environment

Modify `dqn/make_env.py`. MarltasCore's `Actor` and `Evaluator` assume Open-AI gym environment as the interface. You may also need to modify dimensional size settings of observation and action by updating YAML config and the model architecture. Model architecture definitions are written in `dqn/episodic_curioisty.py`, `dqn/model.py`, `dqn/rnn/model.py` and `dqn/cnn/model.py` for episodic curiosity's feature extractor, RND model, RNN-DQN model, and CNN-DQN model respectively.

## Development guide

### How to init your development environment.

Initialize pre-commit hooks.

```bash
pip install yapf isort autoflake pre-commit
pre-commit install
```

### How to run tests

Use `pytest` to test modules.

```bash
pip install pytest
```

```bash
PYTHONPATH=. pytest test
```

## License

MIT
