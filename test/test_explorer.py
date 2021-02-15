import numpy as np

from dqn.explorer import Explorer, SlidingWindowUCB


def test_sliding_window_ucb():
    ucb = SlidingWindowUCB(n_arms=32, window_size=160, eps=0.0)
    uniform = lambda: np.random.random() * 2 - 1.0

    for _ in range(ucb.window_size * 10):
        arm = ucb.select_arm()
        ucb.update(uniform())


def test_epsilon_greedy():
    explorer = Explorer(action_size=10,
                        init_eps=0.4,
                        init_beta=0.5,
                        use_intrinsic_reward=True,
                        use_ucb=False,
                        apply_value_scaling=True)
    uniform = lambda: np.random.random() * 2 - 1.0

    for _ in range(10):
        for _ in range(100):
            # step
            action = explorer.select_action(q_extrinsic=uniform(), q_intrinsic=uniform())
            explorer.on_step(uniform())
        explorer.on_done()


def test_epsilon_ucb():
    explorer = Explorer(action_size=10,
                        init_eps=0.4,
                        init_beta=0.5,
                        use_intrinsic_reward=True,
                        use_ucb=True,
                        apply_value_scaling=True)
    uniform = lambda: np.random.random() * 2 - 1.0

    for _ in range(10):
        for _ in range(100):
            # step
            action = explorer.select_action(q_extrinsic=uniform(), q_intrinsic=uniform())
            explorer.on_step(uniform())
        explorer.on_done()
