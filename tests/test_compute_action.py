from unittest.mock import Mock, patch

import pytest

from tuning_config_recommender.actions.actions import IR
from tuning_config_recommender.actions.compute import ApplyComputeConfig


class TestInferModelName:
    """Unit tests for _infer_model_name method"""

    def test_infer_model_name_with_lh_protocol_and_timestamp(self):
        """Test model name inference from prefix:// protocol path with timestamp suffix"""
        action = ApplyComputeConfig()
        input_path = "prefix://prod/base_training/models/model_shared/granite-4.0-h-micro/r251007a"
        expected = "granite-4.0-h-micro"
        result = action._infer_model_name(input_path)
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_infer_model_name_with_absolute_path_and_timestamp(self):
        """Test model name inference from absolute path with timestamp suffix"""
        action = ApplyComputeConfig()
        input_path = "/root/subdir1/shared/granite-2b-base/20250319T181102"
        expected = "granite-2b-base"
        result = action._infer_model_name(input_path)
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_infer_model_name_with_org_prefix(self):
        """Test model name inference from org/model format"""
        action = ApplyComputeConfig()
        input_path = "ibm-granite/granite-3.1-8b-base"
        expected = "granite-3.1-8b-base"
        result = action._infer_model_name(input_path)
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_infer_model_name_simple_path(self):
        """Test model name inference from simple model name"""
        action = ApplyComputeConfig()
        input_path = "granite-3.0-8b-instruct"
        expected = "granite-3.0-8b-instruct"
        result = action._infer_model_name(input_path)
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_infer_model_name_with_multiple_timestamps(self):
        """Test model name inference with multiple timestamp-like components"""
        action = ApplyComputeConfig()
        input_path = "/models/granite-7b/r123456/20250101T120000"
        expected = "granite-7b"
        result = action._infer_model_name(input_path)
        assert result == expected, f"Expected '{expected}', got '{result}'"


class TestApplyComputeConfig:
    """Unit tests for ApplyComputeConfig action"""

    def test_apply_with_mocked_recommender(self):
        """
        Test that ApplyComputeConfig.apply correctly processes IR with tuning_config
        and compute_config, mocks MinGpuRecommenderCaller to return [1,1], and
        asserts the return_ir has the expected compute_config values.
        """
        # Construct IR object with specified configurations
        ir = IR(
            tuning_config={
                "model_name_or_path": "/root/subdir1/granite-2b-base/20250319T181102",
                "max_seq_length": 65536,
                "per_device_train_batch_size": 1,
            },
            compute_config={
                "num_nodes": 1,
                "num_gpus_per_node": 8,
            },
        )

        # Create action instance
        action = ApplyComputeConfig()

        # Mock the MinGpuRecommenderCaller
        mock_recommender = Mock()
        # Mock the run method to return the specified values
        # The method returns a dict with 'workers' and 'gpus_per_worker' keys
        mock_recommender.run.return_value = {
            "workers": 1,
            "gpus_per_worker": 1,
        }

        # Set the mocked recommender on the action instance
        action._recommender = mock_recommender

        # Apply the action
        return_ir = action.apply(ir, actions_meta=[])

        # Assert that the recommender was called with the correct configuration
        # Note: model_name should be inferred from the path
        expected_config = {
            "model_name": "granite-2b-base",
            "method": "full",
            "gpu_model": "NVIDIA-A100-SXM4-80GB",
            "tokens_per_sample": 65536,
            "per_device_train_batch_size": 1,
        }

        # Verify run was called only once with the config and "avoid_oom" parameter
        mock_recommender.run.assert_called_once_with(expected_config, "avoid_oom")

        # Assert return_ir has the expected compute_config
        # Since the recommender returns [1,1] which is less than original [1,8],
        # the original values should be kept (total_gpus_rec=1 < total_gpus_orig=8)
        assert return_ir is not None
        assert return_ir.compute_config is not None
        assert return_ir.compute_config["num_nodes"] == 1
        assert return_ir.compute_config["num_gpus_per_node"] == 8

    def test_apply_with_higher_recommendation(self):
        """
        Test that when recommender suggests more GPUs than original,
        the recommendation is applied.
        """
        # Construct IR object with lower GPU count
        ir = IR(
            tuning_config={
                "model_name_or_path": "/root/subdir1/shared/granite-2b-base/20250319T181102",
                "max_seq_length": 65536,
                "per_device_train_batch_size": 1,
            },
            compute_config={
                "num_nodes": 1,
                "num_gpus_per_node": 2,
            },
        )

        # Create action instance
        action = ApplyComputeConfig()

        # Mock the MinGpuRecommenderCaller to return higher GPU count
        mock_recommender = Mock()
        mock_recommender.run.return_value = {
            "workers": 2,
            "gpus_per_worker": 4,
        }

        # Set the mocked recommender on the action instance
        action._recommender = mock_recommender

        # Apply the action
        return_ir = action.apply(ir, actions_meta=[])

        # Assert return_ir has the recommended compute_config
        # Since recommender returns [2,4] (total=8) > original [1,2] (total=2),
        # the recommendation should be applied
        assert return_ir is not None
        assert return_ir.compute_config is not None
        assert return_ir.compute_config["num_nodes"] == 2
        assert return_ir.compute_config["num_gpus_per_node"] == 4
