import json
from copy import deepcopy
from pathlib import Path

from loguru import logger

from tuning_config_recommender.actions import IR
from tuning_config_recommender.rule_engine import RuleEngine
from tuning_config_recommender.utils.adapter_utils import (
    build_launch_command,
    prepare_ir_for_accelerate,
    write_yaml_preserving_templates,
)
from tuning_config_recommender.utils.data_processing import get_model_path


class Adapter:
    def execute(self):
        pass


class VanillaAdapter(Adapter):
    def execute(
        self,
        tuning_config,
        compute_config,
        accelerate_config,
        data_config,
        unique_tag,
        skip_estimator=None,
    ):
        try:
            re = RuleEngine()
            re.register_all_inbuilt_actions()
            if hasattr(self, "additional_actions") and self.additional_actions:
                logger.info("Registering additional actions")
                for _, action_cls in self.additional_actions.items():
                    re.register_action(action_cls())
            if skip_estimator:
                re.add_to_actions_meta("skip_estimator")
            model_name_or_path = tuning_config["model_name_or_path"]
            local_model_name_or_path = get_model_path(
                model_name_or_path, unique_tag=unique_tag
            )
            tuning_config["model_name_or_path"] = local_model_name_or_path
            tuning_config["original_model_name_or_path"] = model_name_or_path
            if "tuning_strategy" not in tuning_config:
                tuning_config["tuning_strategy"] = "full"
                if tuning_config.get("peft_method", None) == "lora":
                    tuning_config["tuning_strategy"] = "lora"

            ir = IR(
                tuning_config=tuning_config,
                compute_config=compute_config,
                accelerate_config=accelerate_config,
                tuning_data_config=data_config,
            )
            ir_to_apply, json_patches = re.apply(ir=deepcopy(ir))
            ir_to_apply.tuning_config.pop("tuning_strategy")
            return ir_to_apply, json_patches
        except Exception as e:
            logger.error(f"Error in VanillaAdapter.execute: {str(e)}")
            raise Exception(f"Failed to execute VanillaAdapter: {str(e)}") from e


class FMSAdapter(VanillaAdapter):
    def __init__(self, base_dir: str | Path = "out/fms_final", additional_actions=None):
        self.base_dir = Path(base_dir)
        if not additional_actions:
            additional_actions = []
        self.additional_actions = additional_actions

    def _populate_data_config(self, data_paths: list[str]):
        # NOTE: The assumption is all the data paths are uniform
        # type as in they all are either chat or QA.
        return {
            "dataprocessor": {
                "type": "default",
                "streaming": False,
            },
            "datasets": [
                {
                    "name": "dataset_from_inputs",
                    "data_paths": data_paths,
                    "data_handlers": {},
                }
            ],
        }

    def _resolve_data_paths_in_data_config(self, data_config):
        import glob

        try:
            for dataset in data_config.get("datasets", []):
                dataset["data_paths"] = [
                    p for path in dataset.get("data_paths", []) for p in glob.glob(path)
                ]
            return data_config
        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise FileNotFoundError(f"File not found: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error resolving data paths in data config: {str(e)}")
            raise Exception(f"Failed to resolve data paths: {str(e)}") from e

    def execute(
        self,
        tuning_config,
        compute_config,
        accelerate_config,
        data_config,
        unique_tag,
        paths,
        skip_estimator=None,
    ):
        try:
            if not data_config and not tuning_config.get("training_data_path", None):
                # "paths" = {
                #     "chat_data": "",
                #     "qa_data": "",
                # }
                data_paths = []
                for _, path in paths.items():
                    if "_data" in path:
                        data_paths.append(path)
                data_config = self._populate_data_config(data_paths)
            data_config = self._resolve_data_paths_in_data_config(data_config)
            ir, patches = super().execute(
                tuning_config,
                compute_config,
                accelerate_config,
                data_config,
                unique_tag,
                skip_estimator,
            )

            ir = ir.to_dict()
            target_dir = (self.base_dir / unique_tag).resolve()
            target_dir.mkdir(parents=True, exist_ok=True)

            orig = ir["tuning_config"].pop("original_model_name_or_path", None)
            if orig:
                ir["tuning_config"]["model_name_or_path"] = orig

            ir_clean, dynamic_args = prepare_ir_for_accelerate(ir)
            data_path = target_dir / "tuning_data_config.yaml"
            write_yaml_preserving_templates(
                ir_clean.get("tuning_data_config", {}), data_path
            )

            accel_path = target_dir / "accelerate_config.yaml"
            write_yaml_preserving_templates(
                ir_clean.get("accelerate_config", {}), accel_path
            )

            tuning_config_path = target_dir / "tuning_config.yaml"
            compute_config_path = target_dir / "compute_config.yaml"
            write_yaml_preserving_templates(
                ir_clean.get("tuning_config", {}), tuning_config_path
            )
            write_yaml_preserving_templates(
                ir_clean.get("compute_config", {}), compute_config_path
            )
            launch_cmd = build_launch_command(
                ir_clean, data_path, accel_path, dynamic_args
            )
            serializable_patches = []
            for patch in patches:
                serializable_patches.append(
                    {
                        "json_patch": patch["json_patch"],
                        "comment": str(patch["comment"]),
                    }
                )
            return json.loads(
                json.dumps(
                    {
                        "launch_command": launch_cmd,
                        "paths": {
                            "tuning_config": str(tuning_config_path),
                            "compute_config": str(compute_config_path),
                            "accelerate_config": str(accel_path),
                            "tuning_data_config": str(data_path),
                        },
                        "dict_payload": {
                            "step_config_section": {
                                "tuning_data_config": ir_clean.get(
                                    "tuning_data_config", {}
                                ),
                                "tuning_config": ir_clean.get("tuning_config", {}),
                                "compute_config": ir_clean.get("compute_config", {}),
                                "acceleration_config": ir_clean.get(
                                    "accelerate_config", {}
                                ),
                            }
                        },
                        "patches": patches,
                        "serializable_patches": serializable_patches,
                    },
                    default=str,
                )
            )
        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise FileNotFoundError(f"File not found: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error in FMSAdapter.execute: {str(e)}")
            raise Exception(f"Failed to execute FMSAdapter: {str(e)}") from e
