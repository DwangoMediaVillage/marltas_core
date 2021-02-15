from dqn.cnn.config import CNNConfigBase
from dqn.cnn.model import Model


def get_param_sum(model):
    res = 0
    for p in model.parameters():
        res += p.detach().numpy().sum()
    return res


def test_param():
    model_a = Model(CNNConfigBase())
    model_b = Model(CNNConfigBase())

    assert get_param_sum(model_a) != get_param_sum(model_b)

    x = model_a.get_param()
    model_b.update_param(x)

    assert get_param_sum(model_a) == get_param_sum(model_b)
