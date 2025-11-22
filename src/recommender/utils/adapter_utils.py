import json
import yaml
import re
from pathlib import Path
from typing import Any, Dict, List

DYNAMIC_PATTERN = re.compile(r"^\$\{([A-Za-z0-9_]+)\}$")

def safe_serialize(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [safe_serialize(o) for o in obj]
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return safe_serialize(obj.__dict__)
    return str(obj)


def write_yaml_preserving_templates(obj: Any, path: Path):
    clean_obj = safe_serialize(obj)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean_obj, f, sort_keys=False, allow_unicode=True, width=10000)


def split_static_and_dynamic(cfg: dict):
    static, dynamic = {}, []

    for k, v in cfg.items():
        if isinstance(v, str) and (m := DYNAMIC_PATTERN.match(v)):
            dynamic.append(f"--{k} \"${{{m.group(1)}}}\"")
        else:
            static[k] = v

    return static, dynamic


def fmt_cli_value(v):
    if isinstance(v, bool):
        return "'true'" if v else "'false'"
    if isinstance(v, (int, float)):
        return f"'{v}'"
    if isinstance(v, dict):
        return f"'{json.dumps(v)}'"
    if isinstance(v, (list, tuple)):
        return " ".join(f"'{str(x).lower()}'" for x in v)
    return f"'{v}'"


def prepare_ir_for_accelerate(ir: dict):
    static_dist, dynamic = split_static_and_dynamic(ir.get("dist_config", {}))
    ir["dist_config"] = static_dist
    return ir, dynamic

def build_launch_command(
    ir: Dict[str, Any],
    data_config_path: Path,
    accelerate_config_path: Path,
    dynamic_args: List[str] = None,
) -> str:
    train = ir.get("train_config", {})
    cmd = [
        "accelerate launch",
        f"--config_file {accelerate_config_path}",
        *(dynamic_args or []),
        "-m 'tuning.sft_trainer'",
    ]

    for k, v in ir.get("train_config", {}).items():
        if v is not None and k != "training_data_path":
            cmd.append(f"--{k} {fmt_cli_value(v)}")

    cmd.append(f"--data_config {data_config_path}")

    return " \\\n".join(cmd)
