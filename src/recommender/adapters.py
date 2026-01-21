from copy import deepcopy
from pathlib import Path
from typing import List
from recommender.actions import IR
from recommender.rule_engine import RuleEngine
from recommender.utils.adapter_utils import (
    build_launch_command,
    prepare_ir_for_accelerate,
    write_yaml_preserving_templates,
)
from recommender.utils.data_processing import get_model_path


class Adapter:
    def execute(self):
        pass


class VanillaAdapter(Adapter):
    def execute(
        self, train_config, compute_config, dist_config, data_config, unique_tag
    ):
        re = RuleEngine()
        re.register_all_inbuilt_actions()
        model_name_or_path = train_config["model_name_or_path"]
        local_model_name_or_path = get_model_path(
            model_name_or_path, unique_tag=unique_tag
        )
        train_config["model_name_or_path"] = local_model_name_or_path
        train_config["original_model_name_or_path"] = model_name_or_path

        ir = IR(
            train_config=train_config,
            compute_config=compute_config,
            dist_config=dist_config,
            data_preprocessor=data_config,
        )
        ir_to_apply, json_patches = re.apply(ir=deepcopy(ir))
        return ir_to_apply, json_patches


class FMSAdapter(VanillaAdapter):
    def __init__(self, base_dir: str | Path = "out/fms_final"):
        self.base_dir = Path(base_dir)

    def _populate_data_config(self, data_paths: List[str]):
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

    def execute(
        self, train_config, compute_config, dist_config, data_config, unique_tag, paths
    ):
        if not data_config and not train_config.get("training_data_path", None):
            # "paths" = {
            #     "chat_data": "",
            #     "qa_data": "",
            # }
            data_paths = []
            for name, path in paths.items():
                if "_data" in path:
                    data_paths.append(path)
            data_config = self._populate_data_config(data_paths)
        ir, _ = super().execute(
            train_config, compute_config, dist_config, data_config, unique_tag
        )
        ir = ir.to_dict()
        target_dir = (self.base_dir / unique_tag).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        orig = ir["train_config"].pop("original_model_name_or_path", None)
        if orig:
            ir["train_config"]["model_name_or_path"] = orig

        ir_clean, dynamic_args = prepare_ir_for_accelerate(ir)
        data_path = target_dir / "data_config.yaml"
        write_yaml_preserving_templates(
            ir_clean.get("data_preprocessor", {}), data_path
        )

        accel_path = target_dir / "accelerate_config.yaml"
        write_yaml_preserving_templates(ir_clean.get("dist_config", {}), accel_path)

        tuning_config_path = target_dir / "tuning_config.yaml"
        compute_config_path = target_dir / "compute_config.yaml"
        write_yaml_preserving_templates(
            ir_clean.get("train_config", {}), tuning_config_path
        )
        write_yaml_preserving_templates(
            ir_clean.get("compute_config", {}), compute_config_path
        )
        launch_cmd = build_launch_command(ir_clean, data_path, accel_path, dynamic_args)

        return {
            "data_config": str(data_path),
            "accelerate_config": str(accel_path),
            "launch_command": launch_cmd,
            "tuning_config": tuning_config_path,
            "compute_config": compute_config_path,
            "dict_payload": {
                "step_config_section": {
                    "tuning_data_config": ir_clean.get("data_preprocessor", {}),
                    "tuning_config": ir_clean.get("train_config", {}),
                    "compute_config": ir_clean.get("compute_config", {}),
                    "acceleration_config": ir_clean.get("dist_config", {}),
                }
            },
        }
