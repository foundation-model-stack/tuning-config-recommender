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
    _recommender: "MinGpuRecommenderCaller" = None

    def __init__(self):
        if not skip_autoconf:
            if self._recommender is None:
                logger.debug("No recommender instance set.. creating one")
                self._recommender = MinGpuRecommenderCaller()

    def heuristic_skip(self, ir):
        return skip_autoconf

    def apply(self, ir: IR, actions_meta: list[str]) -> IR:
        if "skip_estimator" in actions_meta:
            self.skip = True
            return
        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return
        # TODO: fast kernels are not supported for some optimizer classes
        # we should either edit this or skip this optimization

        # Test for presence of required configs
        if not ir.compute_config:
            logger.warning("compute_config is not present in IR, skipping compute action")
            self.skip = True
            return

        if not ir.tuning_config:
            logger.warning("tuning_config is not present in IR, skipping compute action")
            self.skip = True
            return

        num_nodes = ir.compute_config.get("num_nodes", 1)
        num_gpus_per_node = ir.compute_config.get("num_gpus_per_node", 8)

        logger.debug(
            f"IR has the configuration workers:{num_nodes} gpus:{num_gpus_per_node}"
        )
        # invoke min_gpu_recommender

        r_model_name = ir.tuning_config.get("model_name_or_path", None)
        if not r_model_name:
            raise Exception(f"model name was not populated in the representation {ir}")

        # Check if tuning method is enabled, if not go for full
        r_method = ir.tuning_config.get("tuning_strategy", "full")

        # set GPU model - for now we assume it's for Vela
        # TODO: obtain some information from the environment to determine which
        # cluster, GPU, etc. are being targeted
        r_gpu = "NVIDIA-A100-SXM4-80GB"

        r_max_seq_length = ir.tuning_config.get("max_seq_length", 2048)
        r_batch_size = ir.tuning_config.get("per_device_train_batch_size", 1)
        configuration = {
                "model_name": r_model_name,
                "method": r_method,
                "gpu_model": r_gpu, #Assume it's Vela
                "tokens_per_sample": r_max_seq_length,
                "per_device_train_batch_size": r_batch_size,
        }

        logger.debug(f"Sending this configuration to min gpu recommender: {configuration}")

        res = self._recommender.run(configuration, "avoid_oom")
        if res["gpus_per_worker"] == -1:
            logger.debug(f"Recommender was not able to issue recommender for {configuration}")
        else:
            logger.debug(f"Recommender recommends - {res['workers']} {res['gpus_per_worker']}; original {num_nodes} {num_gpus_per_node}")
            total_gpus_orig = num_gpus_per_node * num_nodes
            total_gpus_rec = res["gpus_per_worker"] * res["workers"]
            #Since we are trying to avoid OOM, we only replace if the recommendation is higher than original request as this means that
            #the number of gpus was too low to avoid OOM
            if total_gpus_rec > total_gpus_orig:
                logger.debug("Replacing original compute config with recommender's suggestion")
                num_gpus_per_node = res["gpus_per_worker"]
                num_nodes = res["workers"]
            else:
                logger.debug("Recommender's suggestion is lower than original request, so not replacing")
        
        #Update fast_moe section 
        #Needs confirmation from the devs
        

        return_ir = IR(
            compute_config={
                "num_nodes": num_nodes,
                "num_gpus_per_node": num_gpus_per_node,
            },
            type=PatchType.COMPATIBILITY,
            level=PatchLevel.SUGGESTION,
            comment=Comment(
                "compute config for single node configuration"
                if (num_nodes == 1 and num_gpus_per_node == 8)
                else ""
            ),
        )
        self.json_merge_patches.append(return_ir)
        self.skip = True
        return return_ir
