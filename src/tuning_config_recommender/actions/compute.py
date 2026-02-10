from typing import Optional

from loguru import logger

from .actions import IR, Action, Comment, PatchLevel, PatchType

try:
    from fm_training_estimator.regressor.min_gpu.recommender import (
        MinGpuRecommenderCaller,
    )

    skip_autoconf = False
except ImportError:
    skip_autoconf = True
    logger.warning(
        "autoconf is not installed therefore corresponding features will not be used"
    )


class ApplyComputeConfig(Action):
    """Action to apply compute configuration recommendations using GPU estimator."""

    # Constants
    DEFAULT_GPU_MODEL = "NVIDIA-A100-SXM4-80GB"
    DEFAULT_MAX_SEQ_LENGTH = 2048
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_TUNING_METHOD = "full"
    RECOMMENDER_FAILURE_CODE = -1
    RECOMMENDER_MODE = "avoid_oom"

    _recommender: MinGpuRecommenderCaller | None = None

    def __init__(self):
        if not skip_autoconf:
            if self._recommender is None:
                logger.debug("No recommender instance set.. creating one")
                self._recommender = MinGpuRecommenderCaller()  # type: ignore

    def heuristic_skip(self, ir):
        return skip_autoconf

    def _infer_model_name(self, m: str) -> str:
        """
        Method to infer model name from model_name_or_path parameter in the IR
        :param m: model_name_or_path parameter value from the IR
        :type m: str
        :return: the model name to pass to the min gpu recommender.
        This has been extracted from the IR and mapped to a model name in the Min GPU recommender database.
        The mapping will return the input if no match is found.
        :rtype: str

        Examples:
            'lh://prod/base_training/models/model_shared/granite-4.0-h-micro/r251007a' -> 'granite-4.0-h-micro'
            '/home/shared/granite-2b-base/20250319T181102' -> 'granite-2b-base'
            'ibm-granite/granite-3.1-8b-base' -> 'granite-3.1-8b-base'
        """
        import re

        logger.debug(f"Model path received: {m}")

        # Split by '/' to get path components
        components = m.split('/')

        # Filter out empty components
        components = [c for c in components if c]

        if not components:
            logger.warning(f"No valid components found in model path: {m}")
            return m

        # Pattern to identify timestamp-like suffixes (e.g., r251007a, 20250319T181102)
        # These typically start with 'r' followed by digits, or are pure date/timestamp formats
        timestamp_pattern = re.compile(r'^(r\d+[a-z]?|\d{8}T\d{6}|\d{14})$', re.IGNORECASE)

        # Work backwards through components to find the model name
        # Skip the last component if it looks like a timestamp/tag
        for i in range(len(components) - 1, -1, -1):
            component = components[i]

            # Skip if it matches timestamp pattern
            if timestamp_pattern.match(component):
                logger.debug(f"Skipping timestamp-like component: {component}")
                continue

            # Skip protocol prefixes (e.g., 'lh:', 'http:', 'https:')
            if component.endswith(':'):
                logger.debug(f"Skipping protocol component: {component}")
                continue

            # This should be the model name
            logger.debug(f"Inferred model name: {component}")
            return component

        # Fallback: return the last non-empty component
        result = components[-1]
        logger.debug(f"Fallback to last component: {result}")
        return result

    def _validate_required_configs(self, ir: IR) -> bool:
        """Validate that required configs are present in IR.

        Args:
            ir: Intermediate representation to validate

        Returns:
            bool: True if valid, False otherwise (sets self.skip and logs warning)
        """
        if not ir.compute_config:
            logger.warning("compute_config is not present in IR, skipping compute action")
            self.skip = True
            return False

        if not ir.tuning_config:
            logger.warning("tuning_config is not present in IR, skipping compute action")
            self.skip = True
            return False

        return True

    def _build_recommender_config(self, ir: IR) -> dict:
        """Build configuration dict for the GPU recommender.

        Args:
            ir: Intermediate representation with tuning config

        Returns:
            dict: Configuration for recommender

        Raises:
            ValueError: If model name cannot be determined
        """
        # Type guard - we know tuning_config is not None due to validation
        assert ir.tuning_config is not None

        # Extract model name with fallback
        model_name = ir.tuning_config.get("model_name_or_path") or ir.tuning_config.get("hf_path")
        if not model_name:
            raise ValueError(f"model name was not populated in the representation {ir}")

        # Infer the actual model name from the path
        inferred_model_name = self._infer_model_name(model_name)

        return {
            "model_name": inferred_model_name,
            "method": ir.tuning_config.get("tuning_strategy", self.DEFAULT_TUNING_METHOD),
            "gpu_model": self.DEFAULT_GPU_MODEL,  # TODO: Make configurable based on environment
            "tokens_per_sample": ir.tuning_config.get("max_seq_length", self.DEFAULT_MAX_SEQ_LENGTH),
            "per_device_train_batch_size": ir.tuning_config.get("per_device_train_batch_size", self.DEFAULT_BATCH_SIZE),
        }

    def _apply_recommendation(
        self,
        config: dict,
        current_nodes: int,
        current_gpus: int
    ) -> tuple[int, int]:
        """Apply GPU recommender and decide whether to use recommendation.

        Args:
            config: Configuration for recommender
            current_nodes: Current number of nodes
            current_gpus: Current GPUs per node

        Returns:
            tuple[int, int]: (num_nodes, num_gpus_per_node) to use
        """
        logger.debug(f"Sending configuration to min gpu recommender: {config}")

        result = self._recommender.run(config, self.RECOMMENDER_MODE)  # type: ignore

        # Early return if recommender failed
        if result["gpus_per_worker"] == self.RECOMMENDER_FAILURE_CODE:
            logger.debug(f"Recommender was not able to issue recommendation for {config}")
            return current_nodes, current_gpus

        recommended_nodes = result["workers"]
        recommended_gpus = result["gpus_per_worker"]

        logger.debug(
            f"Recommender recommends - {recommended_nodes} nodes, {recommended_gpus} GPUs; "
            f"original {current_nodes} nodes, {current_gpus} GPUs"
        )

        total_gpus_original = current_gpus * current_nodes
        total_gpus_recommended = recommended_gpus * recommended_nodes

        # Only apply recommendation if it suggests more GPUs (to avoid OOM)
        if total_gpus_recommended > total_gpus_original:
            logger.debug("Replacing original compute config with recommender's suggestion")
            return recommended_nodes, recommended_gpus
        else:
            logger.debug("Recommender's suggestion is lower than original request, keeping original")
            return current_nodes, current_gpus

    def _generate_comment(self, num_nodes: int, num_gpus_per_node: int) -> str:
        """Generate appropriate comment for the compute configuration.

        Args:
            num_nodes: Number of nodes
            num_gpus_per_node: GPUs per node

        Returns:
            str: Comment message
        """
        if num_nodes == 1 and num_gpus_per_node == 8:
            return "compute config for single node configuration"
        return ""

    def apply(self, ir: IR, actions_meta: list[str] = None) -> IR:
        """Apply compute configuration recommendations.

        Args:
            ir: Intermediate representation
            actions_meta: List of action metadata flags

        Returns:
            IR with updated compute config, or None if skipped
        """
        if actions_meta is None:
            actions_meta = []

        # Early returns for skip conditions
        if "skip_estimator" in actions_meta:
            self.skip = True
            return ir

        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return ir

        # TODO: fast kernels are not supported for some optimizer classes
        # we should either edit this or skip this optimization

        # Validate required configurations
        if not self._validate_required_configs(ir):
            return ir

        # Type guards - we know these are not None due to validation
        assert ir.compute_config is not None

        # Extract current compute config
        num_nodes = ir.compute_config.get("num_nodes", 1)
        num_gpus_per_node = ir.compute_config.get("num_gpus_per_node", 8)

        logger.debug(f"IR has configuration: {num_nodes} nodes, {num_gpus_per_node} GPUs")

        # Build and apply recommendation
        try:
            recommender_config = self._build_recommender_config(ir)
            num_nodes, num_gpus_per_node = self._apply_recommendation(
                recommender_config, num_nodes, num_gpus_per_node
            )
        except ValueError as e:
            logger.error(f"Failed to build recommender config: {e}")
            self.skip = True
            return ir

        # Build result IR
        return_ir = IR(
            compute_config={
                "num_nodes": num_nodes,
                "num_gpus_per_node": num_gpus_per_node,
            },
            type=PatchType.COMPATIBILITY,
            level=PatchLevel.SUGGESTION,
            comment=self._generate_comment(num_nodes, num_gpus_per_node),
        )

        self.json_merge_patches.append(return_ir)
        self.skip = True
        return return_ir
