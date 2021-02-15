from test.dammy_env import DammyEnv

from dqn.cnn.config import CNNConfigBase
from dqn.cnn.evaluator import Evaluator
from dqn.evaluator import EvaluationRequest


def test_evaluator():
    evaluator = Evaluator(config=CNNConfigBase(), make_env_func=lambda config: DammyEnv(max_step=10))
    result = evaluator.evaluate(EvaluationRequest(global_step=1))
    assert result.global_step == 1
    assert isinstance(result.walltime, float)
    assert isinstance(result.reward_sum, float)
    assert isinstance(result.episode_len, float)
