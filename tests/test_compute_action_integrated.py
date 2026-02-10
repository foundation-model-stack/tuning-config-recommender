import pytest

from tuning_config_recommender.actions.actions import IR
from tuning_config_recommender.actions.compute import ApplyComputeConfig


def test_min_gpu_recommender_caller():
    action = ApplyComputeConfig()
    assert action._recommender is not None

@pytest.fixture
def get_compute_action() -> ApplyComputeConfig:
    yield ApplyComputeConfig()

def test_apply_with_lower_recommendation(get_compute_action):
    """
    Test that when recommender suggests more GPUs than original,
    the recommendation is applied.
    """
    # Construct IR object with lower GPU count
    ir: IR = IR(
        tuning_config={
            "hf_path": "ibm-granite/granite-3.1-8b-base",
            "max_seq_length": 1024,
            "per_device_train_batch_size": 1,
        },
        compute_config={
            "num_nodes": 4,
            "num_gpus_per_node": 8,
        },
    )
    action = get_compute_action
    return_ir = action.apply(ir)

    assert return_ir is not None
    assert return_ir.compute_config is not None
    assert return_ir.compute_config["num_nodes"] == 4
    assert return_ir.compute_config["num_gpus_per_node"] == 8

def test_apply_with_higher_recommendation(get_compute_action):
    """
    Test that when recommender suggests more GPUs than original,
    the recommendation is applied.
    """
    # Construct IR object with lower GPU count
    ir: IR = IR(
        tuning_config={
            "hf_path": "/home/shared/granite-2b-base/20250319T181102",
            "max_seq_length": 32768,
            "per_device_train_batch_size": 8,
        },
        compute_config={
            "num_nodes": 1,
            "num_gpus_per_node": 2,
        },
    )
    action = get_compute_action
    return_ir = action.apply(ir)

    assert return_ir is not None
    assert return_ir.compute_config is not None
    assert return_ir.compute_config["num_nodes"] == 4
    assert return_ir.compute_config["num_gpus_per_node"] == 8
