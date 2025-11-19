from recommender.actions import IR
from recommender.utils.data_processing import get_model_path
from recommender.rule_engine import RuleEngine
from copy import deepcopy
from pathlib import Path
from recommender.utils.adapter_utils import (
    safe_serialize,
    write_yaml_preserving_templates,
    build_launch_command,
)


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
        local_model_name_or_path = get_model_path(model_name_or_path, unique_tag=unique_tag)
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

    def _to_target(self, ir, patches=None, tag=None):
        ir = ir.to_dict()
        target_dir = (self.base_dir / tag).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        orig = ir["train_config"].pop("original_model_name_or_path", None)
        if orig:
            ir["train_config"]["model_name_or_path"] = orig

        data_path = target_dir / "data_config.yaml"
        write_yaml_preserving_templates(ir.get("data_preprocessor", {}), data_path)

        accel_path = target_dir / "accelerate_config.yaml"
        write_yaml_preserving_templates(ir.get("dist_config", {}), accel_path)

        launch_cmd = build_launch_command(ir, data_path, accel_path)

        print(f"[FMSAdapter] Created configs under {target_dir}")
        return {
            "data_config": str(data_path),
            "accelerate_config": str(accel_path),
            "launch_command": launch_cmd,
        }

    def run(self, train_config, compute_config=None, dist_config=None, data_config=None, unique_tag=None):
        ir, patches = self.execute(
            train_config,
            compute_config or {},
            dist_config or {},
            data_config or {},
            unique_tag,
        )
        return self._to_target(ir, patches, tag=unique_tag)
