import json
import yaml
from pathlib import Path
from typing import Any, Dict, List


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
    path.parent.mkdir(parents=True, exist_ok=True)

    def recurse(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "template" and isinstance(v, str):
                    o[k] = json.dumps(v)
                else:
                    recurse(v)
        elif isinstance(o, list):
            for item in o:
                recurse(item)
        return o

    clean_obj = recurse(safe_serialize(obj))
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean_obj, f, sort_keys=False, allow_unicode=True, width=10000)


def build_launch_command(ir: Dict[str, Any], data_config_path: Path) -> str:
    dist, train = ir.get("dist_config", {}), ir.get("train_config", {})
    cmd = ["accelerate launch"]

    def fmt(v):
        s = lambda x: str(x).lower() if str(x).lower() in ("true", "false") else str(x)
        if isinstance(v, (list, tuple)): 
            return " ".join(s(x) for x in v)
        if isinstance(v, dict): 
            return f"'{json.dumps(v)}'"
        return s(v)

    def add_args(config: Dict[str, Any]):
        for k, v in config.items():
            if v is None or k == "training_data_path":
                continue
            # assuming only fsdp_config dict
            if k == "fsdp_config" and isinstance(v, dict):
                for subk, subv in v.items():
                    cmd.append(f"--{subk} {fmt(subv)}")
            else:
                cmd.append(f"--{k} {fmt(v)}")

    add_args(dist)
    cmd += ["-m tuning.sft_trainer"]
    add_args(train)
    cmd.append(f"--data_config {data_config_path}")

    return " \\\n".join(cmd)


