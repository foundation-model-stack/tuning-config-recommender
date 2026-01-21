import json
import os
from pathlib import Path

import pandas as pd
import yaml
from tuning_config_recommender.utils.kb_table import query_kb

script_dir = Path(__file__).resolve().parent


def is_model_type_moe(model_name_or_path: str) -> bool:
    """Checks if the granite model given is MoE"""

    if os.path.isdir(model_name_or_path):
        with open(f"{model_name_or_path}/config.json", encoding="utf-8") as f:
            config = json.load(f)

    moe_tags = ["granitemoe", ""]

    if "model_type" in config and (
        config["model_type"] in moe_tags or "moe" in config["model_type"]
    ):
        return True
    if "architectures" in config and (
        config["architectures"][0].lower() in moe_tags
        or "moe" in config["architectures"][0].lower()
    ):
        return True
    if "num_experts_per_tok" in config and config["num_experts_per_tok"] > 0:
        return True
    return False


def find_best_row(df, target_length, default_value=None):
    # Step 1: Exact match
    exact_match = df[df["model_max_length"] == target_length]
    if not exact_match.empty:
        return exact_match.iloc[0]

    # Step 2: Nearest smaller value
    smaller_matches = df[df["model_max_length"] < target_length]
    if not smaller_matches.empty:
        nearest = smaller_matches.sort_values(
            by="model_max_length", ascending=False
        ).iloc[0]
        return nearest

    # Step 3: Default
    return default_value


def use_kb_for_batch_size(user_input: dict):
    """Use the knowledge base to determine the optimal batch size"""
    training_run_data_path = (
        script_dir.parent / "knowledge_base" / "tuning_run_data.csv"
    )
    df = pd.read_csv(training_run_data_path)

    model_name_or_path = str(user_input.get("model_name_or_path", ""))
    tuning_strategy = user_input.get("tuning_strategy", "")
    max_seq_length = user_input.get("max_seq_length", 2048)

    try:
        model_name_or_path = model_name_or_path.split("/")[-2]
    except Exception:
        pass

    filtered = df[
        (df["model_name"] == model_name_or_path) & (df["method"] == tuning_strategy)
    ]

    # if match is None:
    if filtered.empty:
        if "instruct" in model_name_or_path:
            model_name_or_path = model_name_or_path.replace("instruct", "base")
        elif "base" in model_name_or_path:
            model_name_or_path = model_name_or_path.replace("base", "instruct")
            filtered = df[
                (df["model_name"] == model_name_or_path)
                & (df["method"] == tuning_strategy)
            ]

    match = find_best_row(filtered, max_seq_length)

    batch_size_configs = {}
    if match is not None:
        batch_size_configs.update(
            {
                "per_device_train_batch_size": int(
                    match.get("per_device_train_batch_size", 1)
                ),
                "model_max_length": int(match.get("model_max_length", 2048)),
                "number_gpus": int(match.get("number_gpus", 16)),
            }
        )
    return batch_size_configs


def fetch_from_knowledge_base(model_name_or_path: str, kb_section):
    model_name = (
        model_name_or_path.split("/")[-1]
        if "/" in model_name_or_path
        else model_name_or_path
    )
    return query_kb(model_name, kb_section)


def get_model_config(model_name_or_path: str):
    with open(f"{model_name_or_path}/config.json", encoding="utf-8") as f:
        config = json.load(f)
    return config
