from recommender.utils.train_config import is_model_type_moe
from recommender.utils.train_config import get_model_config
import math
from recommender.utils.train_config import (
    use_kb_for_batch_size,
)
from .actions import IR, Action, PatchLevel, PatchType, Comment
from recommender.constants import (
    DEFAULT_NUM_NODES,
    DEFAULT_NUM_GPUS_PER_NODE,
)

class ApplyDistributedTraining(Action):

    def apply(self, ir: IR) -> IR:
        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return
        comment = Comment()

        num_nodes = int(ir.compute_config.get("num_nodes", DEFAULT_NUM_NODES))
        num_gpus_per_node = int(ir.compute_config.get("num_gpus_per_node", DEFAULT_NUM_GPUS_PER_NODE))
        num_processes = num_nodes * num_gpus_per_node

        fsdp_sharding_strategy = "FULL_SHARD"
        if ir.compute_config.get("num_nodes", None):
            if int(ir.compute_config.get("num_nodes")) > 1:
                fsdp_sharding_strategy = "HYBRID_SHARD"
                comment.add(
                    f"fsdp_sharding_strategy as {fsdp_sharding_strategy}"
                    "improves throughput reducing inter-node communication."
                    "However use FULL_SHARD if you hit OOM."
                )

        fsdp_state_dict_type = "FULL_STATE_DICT"
        if is_model_type_moe(ir.train_config.get("model_name_or_path")):
            fsdp_state_dict_type = "SHARDED_STATE_DICT"
            comment.add(f"SHARDED_STATE_DICT is needed for compatibility")

        data = {
            "num_processes": num_processes,
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "FSDP",
            "fsdp_config": {
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "fsdp_backward_prefetch": "BACKWARD_PRE",
                "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
                "fsdp_forward_prefetch": False,
                "fsdp_offload_params": False,
                "fsdp_sharding_strategy": fsdp_sharding_strategy,
                "fsdp_state_dict_type": fsdp_state_dict_type,
                "fsdp_cpu_ram_efficient_loading": True,
                "fsdp_sync_module_states": True,
            },
            "machine_rank": "${RANK}",
            "num_machines": "${WORLD_SIZE}",
            "rdzv_backend": "static",
            "same_network": True,
            "main_process_ip": "${MASTER_ADDR}",
            "main_process_port": "${MASTER_PORT}",
        }
        ir = IR(
            dist_config=data,
            type=PatchType.COMPATIBILITY,
            level=PatchLevel.MANDATORY,
            comment=comment,
        )
        self.json_merge_patches.append(ir)
        self.skip = True
        return ir


class ApplyGradientCheckpointing(Action):

    def apply(self, ir: IR) -> IR:
        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return
        gradient_checkpointing = True
        gradient_checkpointing_kwargs = '{"use_reentrant": true}'
        tuning_strategy = ir.train_config.get("tuning_strategy")
        comment = Comment(
            "gradient checkpointing provides memory savings which can translate to throughput by increasing the batchsize"
        )
        if "alora" in tuning_strategy:
            gradient_checkpointing = False
            gradient_checkpointing_kwargs = None
            comment.add("Gradient checkpointing is not supported for ALoRA.")
        ir = IR(
            train_config={
                "gradient_checkpointing": gradient_checkpointing,
                "gradient_checkpointing_kwargs": gradient_checkpointing_kwargs,
            },
            type=PatchType.COMPATIBILITY,
            level=PatchLevel.MANDATORY,
            comment=comment,
        )
        self.json_merge_patches.append(ir)
        self.skip = True
        return ir


class ApplyLoRAConfig(Action):
    def heuristic_skip(self, ir):
        if (
            ir.train_config.get("tuning_strategy") == "lora"
            or ir.train_config.get("peft_method", None) == "lora"
        ):
            return False
        return True

    def apply(self, ir: IR) -> IR:
        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return

        # TODO: ideally this data has to be prepared
        # looking at the model and identify
        # best lora practices
        data = {
            "peft_method": "lora",
            "lora_alpha": 8,
            "lora_dropout": 0.1,
            "r": 8,
            "target_modules": "all-linear",
            "modules_to_save": ["lm_head", "embed_token"],
        }
        ir = IR(
            train_config=data,
            type=PatchType.COMPATIBILITY,
            level=PatchLevel.MANDATORY,
            comment=Comment("LoRA config is added since you are using LoRA training."),
        )
        self.json_merge_patches.append(ir)
        self.skip = True
        return ir


class ApplyMoEOptimization(Action):

    def _get_num_experts(self, model_name_or_path: str) -> int:
        config = get_model_config(model_name_or_path)
        num_local_experts = config.get("num_local_experts", None)
        num_experts = config.get("num_experts", None)
        return num_local_experts or num_experts

    def heuristic_skip(self, ir):
        if self._get_num_experts(ir.train_config.get("model_name_or_path")):
            return False
        return True

    def apply(self, ir: IR) -> IR:
        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return

        # data variables
        fast_moe = True

        num_experts = self._get_num_experts(ir.train_config.get("model_name_or_path"))
        num_nodes = ir.compute_config.get("num_nodes", 1)
        num_gpus_per_node = ir.compute_config.get("num_gpus_per_node", None)
        if num_gpus_per_node:
            fast_moe = math.gcd(num_experts, num_gpus_per_node * num_nodes)

        return_ir = IR(
            train_config={"fast_moe": fast_moe},
            type=PatchType.SYSTEM_PERFORMANCE,
            level=PatchLevel.SUGGESTION,
            comment=Comment("fast_moe option gives 1.5 to 4X speedup."),
        )
        self.json_merge_patches.append(return_ir)
        self.skip = True
        return return_ir


class ApplyOptimalBatchSize(Action):

    def apply(self, ir: IR) -> IR:
        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return

        input_dict = {
            "model_name_or_path": ir.train_config["model_name_or_path"],
            "tuning_strategy": ir.train_config["tuning_strategy"],
            "max_seq_length": ir.train_config.get("max_seq_length", 2048),
        }
        output_dict = use_kb_for_batch_size(input_dict)
        return_ir = IR(
            train_config={
                "per_device_train_batch_size": output_dict.get(
                    "per_device_train_batch_size", 1
                ),
                "max_seq_length": output_dict.get("model_max_length", 2048),
            },
            type=PatchType.COMPATIBILITY,
            effect=[
                PatchType.MODEL_QUALITY,
                PatchType.SYSTEM_PERFORMANCE,
                PatchType.COMPATIBILITY,
            ],
            level=PatchLevel.SUGGESTION,
            comment=Comment(
                "per_device_train_batch_size is modified to best use the GPU resources and not hit OOM."
            ),
        )
        self.json_merge_patches.append(return_ir)
        self.skip = True
        return return_ir


class ApplyFastKernelsOptimization(Action):
    supported_model_archs = [
        "GraniteForCausalLM",
        "GraniteMoeForCausalLM",
        "GPTBigCodeForCausalLM",
        "MixtralForCausalLM",
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "GraniteMoeSharedForCausalLM",
        "GraniteMoeHybridForCausalLM",
    ]

    def heuristic_skip(self, ir):
        config = get_model_config(ir.train_config["model_name_or_path"])
        if config.get("architectures", [""])[0] in self.supported_model_archs:
            return False
        return True

    def apply(self, ir: IR) -> IR:
        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return
        # TODO: fast kernels are not supported for some optimizer classes
        # we should either edit this or skip this optimization
        return_ir = IR(
            train_config={"fast_kernels": ["True", "True", "True"]},
            type=PatchType.SYSTEM_PERFORMANCE,
            level=PatchLevel.SUGGESTION,
            comment=Comment(
                "Provides a throughput boosts of 10%. This flags replaces RoPE, cross entropy loss and RMS layernorm with faster kernel impls."
            ),
        )
        self.json_merge_patches.append(return_ir)
        self.skip = True
        return return_ir


class ApplyTrainingOptimization(Action):

    def apply(self, ir: IR) -> IR:
        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return
        # TODO: fast kernels are not supported for some optimizer classes
        # we should either edit this or skip this optimization
        return_ir = IR(
            train_config={"padding_free": "huggingface", "use_flash_attn": True},
            type=PatchType.SYSTEM_PERFORMANCE,
            level=PatchLevel.SUGGESTION,
            comment=Comment(
                "padding_free with flash_attention provides throughput boost and memory savings."
            ),
        )
        self.json_merge_patches.append(return_ir)
        self.skip = True
        return return_ir
