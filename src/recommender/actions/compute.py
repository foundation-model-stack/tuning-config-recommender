from autoconf.utils.config_mapper import map_valid_model_name
from fm_training_estimator.regressor.min_gpu.recommender import MinGpuRecommenderCaller
from loguru import logger

from .actions import IR, Action, Comment, PatchLevel, PatchType


class ApplyComputeConfig(Action):

    _recommender:MinGpuRecommenderCaller = None
    def __init__(self):
        if self._recommender == None:
            logger.debug(f"No recommender instance set.. creating one")
            self._recommender = MinGpuRecommenderCaller()
    
    def _infer_model_name(self, m:str) -> str:
        """
        Method to infer model name from model_name_or_path parameter in the IR
        :param m: model_name_or_path parameter value from the IR
        :type m: str
        :return: the model name to pass to the min gpu recommender.
        This has been extracted from the IR and mapped to a model name in the Min GPU recommender database. 
        The mapping will return the input if no match is found.
        :rtype: str
        """
        logger.debug(f"Model path received: {m}")
        #When a unique tag is given, the model path will contain an additional path component at the end
        #Thus we will start from the last path component and work backwards for atleast one more mapping
        for _ in range(2):
            components = m.rpartition("/")
            logger.debug(f"Model path split into: {components}")
            r_name = map_valid_model_name(components[2])
            if (r_name == components[2]):
                m = components[0]
                continue
            else:
                break
        logger.debug(f"Stopped at result: {r_name}")

        return r_name

    def apply(self, ir: IR) -> IR:
        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return
        # TODO: fast kernels are not supported for some optimizer classes
        # we should either edit this or skip this optimization
        num_nodes = ir.compute_config.get("num_nodes", 1)
        num_gpus_per_node = ir.compute_config.get("num_gpus_per_node", 8)

        logger.debug(f"IR has the configuration workers:{num_nodes} gpus:{num_gpus_per_node}")
        # invoke min_gpu_recommender
        
        r_model_name = ir.train_config.get("model_name_or_path", None)
        if not r_model_name:
            raise Exception(f"model name was not populated in the representation {ir}")

        #Find a valid mapping for the model name
        r_model_name = self._infer_model_name(r_model_name)

        #Check if tuning method is enabled, if not go for full
        r_method = ir.train_config.get("tuning_strategy", "full")

        #set GPU model - for now we assume it's for Vela
        #TODO: obtain some information from the llm.build space to determine which
        #cluster, GPU, etc. are being targeted
        r_gpu = "NVIDIA-A100-SXM4-80GB"

        #set batch size - this is tricky because we get the per-device batch size from IR
        #while the recommender model uses batch size across gpus.
        #Current solution - test every #GPU in 1,2,4,8 and pick the smallest one that we get
        #recommendation for
        r_max_seq_length = ir.train_config.get("max_seq_length", 2048)
        r_batch_size = ir.train_config.get("per_device_batch_size", 1)

        for r_num_gpu in [1,2,4,8]:
            configuration = {
                "model_name": r_model_name,
                "method": r_method,
                "gpu_model": r_gpu, #Assume it's Vela
                "tokens_per_sample": r_max_seq_length,
                "batch_size": r_batch_size*r_num_gpu,
                "gpus_per_worker": r_num_gpu,
                "model_version": "2.0.0"
            }
            logger.debug(f"Sending this configuration to min gpu recommender: {configuration}")
            res = self._recommender.run(configuration, "min_gpu")
            if res["gpus_per_worker"] == -1:
                logger.debug(f"Recommender was not able to issue recommender for {configuration}")
                continue
            else:
                num_gpus_per_node = res["gpus_per_worker"]
                num_nodes = res["workers"]
                logger.debug(f"recommender returned configuration workers:{num_nodes} gpus:{num_gpus_per_node}")
                break

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
