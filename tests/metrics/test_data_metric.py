from a4s_eval.metric_registry.data_metric_registry import data_evaluator_registry


def test_registration():
    assert len(list(iter(data_evaluator_registry))) >= 1
