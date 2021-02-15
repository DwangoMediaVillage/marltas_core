import pprint
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from dqn.explorer import Explorer
from dqn.make_env import make_env
from dqn.rnn.config import RNNConfigBase
from dqn.rnn.learner import Learner
from dqn.rnn.policy import Policy
from dqn.rollout_logger import RolloutLogger


@dataclass
class RolloutConfig:
    out: Path
    render: bool
    eps: float
    num_episode: int
    num_process: int
    snap: Optional[Path] = None


def rollout(process_index: int, rollout_config: RolloutConfig, config: RNNConfigBase) -> None:
    """Produce episode data using epsilon-greedy.
    Args:
        process_index: Index of process.
        rollout_config: Configuration of rollout.
        config: Configuration of DQN agent.
    """
    assert rollout_config.out.exists()
    if process_index == 0:
        pprint.pprint(asdict(rollout_config))
        pprint.pprint(asdict(config))

    # init policy
    policy = Policy(config=config)

    if rollout_config.snap:
        assert rollout_config.snap.exists()
        learner = Learner(config=config)
        learner.load_online_model(rollout_config.snap)
        policy.update_model_param(learner.get_model_param(), only_online_model=True)

    # init explorer
    explorer = Explorer(action_size=config.model.action_size,
                        init_eps=rollout_config.eps,
                        init_beta=config.intrinsic_reward.reward_ratio,
                        use_intrinsic_reward=False,
                        use_ucb=False,
                        apply_value_scaling=config.apply_value_scaling)

    # init rollout logger
    rollout_logger = RolloutLogger(out_dir=rollout_config.out,
                                   render=rollout_config.render,
                                   filename_header=str(process_index))

    # init env
    env = make_env(config.env)

    render = lambda: env.render(mode='rgb_array')

    for i in range(rollout_config.num_episode):
        print(f"process: {process_index} # episode = {i}")
        obs = env.reset()
        done = False
        state = [policy.model.get_init_state()]
        rollout_logger.on_reset(obs, render() if rollout_config.render else None)

        while not done:
            prediction, intrinsic_reward, state = policy.infer([obs], state)
            q_e, q_i = prediction.as_numpy_tuple()
            action = explorer.select_action(q_e[0])
            obs, reward, done, info = env.step(action)
            rollout_logger.on_step(action=action,
                                   q_e=q_e[0].tolist(),
                                   q_i=q_i[0].tolist() if q_i is not None else None,
                                   intrinsic_reward=intrinsic_reward[0] if intrinsic_reward is not None else None,
                                   obs=obs,
                                   reward=reward,
                                   info=info,
                                   done=done,
                                   image_frame=render() if rollout_config.render else None)


def main(rollout_config: RolloutConfig, config: RNNConfigBase) -> None:
    """Kick `rollout` using Multiprocessing"""

    if rollout_config.num_process == 1:
        # single process
        rollout(process_index=0, rollout_config=rollout_config, config=config)
    else:
        # multi process run
        import functools
        import multiprocessing
        with multiprocessing.Pool(processes=rollout_config.num_process) as pool:
            [
                p for p in pool.imap_unordered(functools.partial(rollout, rollout_config=rollout_config, config=config),
                                               range(rollout_config.num_process))
            ]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate episode data by trained DQN agent.")
    parser.add_argument("out", type=Path, help="Directory to put generated files.")
    parser.add_argument("--config", type=Path, help="DQN configuration YAML file path.")
    parser.add_argument("--snap", type=Path, help="Snapfile of online Q-Network.")
    parser.add_argument("--render", action='store_true', help="Generating mp4 image or not.")
    parser.add_argument("--eps", type=float, default=0.02, help="Epsilon-greedy parameter.")
    parser.add_argument("--num_episode", type=int, default=1, help="Number of episode for each process.")
    parser.add_argument("--num_process", type=int, default=1, help="Number of process running in parallel.")
    args = parser.parse_args()

    # load config
    config = RNNConfigBase.load_from_yaml(yaml_path=args.config,
                                          allow_illegal_keys=True) if args.config else RNNConfigBase()

    # using CPU
    config.learner.gpu_id = None

    rollout_config = RolloutConfig(out=args.out,
                                   render=args.render,
                                   eps=args.eps,
                                   num_episode=args.num_episode,
                                   num_process=args.num_process,
                                   snap=args.snap)

    # creat directory
    rollout_config.out.mkdir(exist_ok=True)

    main(rollout_config=rollout_config, config=config)
