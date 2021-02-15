from test.dammy_env import DammyEnv

from dqn.evaluator import EvaluationRequest
from dqn.rnn.config import EvaluatorConfig, RNNConfigBase
from dqn.rnn.evaluator import Evaluator


def test_evaluator():
    evaluator = Evaluator(
        config=RNNConfigBase(evaluator=EvaluatorConfig(custom_metric_keys=['score'], custom_metric_types=['last'])),
        make_env_func=lambda c: DammyEnv(max_step=10))
    result = evaluator.evaluate(EvaluationRequest(global_step=1))
    assert result.global_step == 1
    assert isinstance(result.walltime, float)
    assert isinstance(result.reward_sum, float)
    assert isinstance(result.episode_len, float)
    assert 'score' in result.custom_metrics
