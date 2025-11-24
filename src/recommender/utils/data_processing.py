import os
import re
import csv
import json
import pandas as pd
from loguru import logger
from pathlib import Path
import shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def extract_data_from_general_file(file_path) -> dict:
    """Data extraction function from json/jsonl/parquet/arrow files"""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif ext == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    elif ext == ".csv":
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
        data = df.to_dict(orient="records")
    elif ext == ".arrow":
        try:
            data = load_dataset("arrow", data_files=file_path)
        except Exception as e:
            logger.error(f"Failed to load Arrow file: {e}")
    else:
        logger.error("Unsupported file format")

    return data


def maybe_is_a_hf_dataset_id(training_data_path: str) -> bool:
    return len(training_data_path.split("/")) == 2


def pick_train_split(dataset) -> str:
    """Choose a split containing 'train' substring"""
    splits = list(dataset.keys())
    if not splits:
        raise ValueError("Dataset has no splits.")
    # substring match with 'train'
    train_like = [s for s in splits if "train" in s.lower()]
    if train_like:
        return train_like[0]

    return splits[0]


def load_training_data(training_data_path: str) -> dict:
    """Load and validate training data based on training_data_path."""
    _dataset_cache = {}

    # Check if path is a file
    if os.path.isfile(training_data_path):
        data = extract_data_from_general_file(training_data_path)
        return data

    # Check if path is a folder
    elif os.path.isdir(training_data_path) or maybe_is_a_hf_dataset_id:
        try:
            dataset = load_dataset(training_data_path)
            split=pick_train_split(dataset)
            data = [dict(example) for example in dataset[split]]
            return data
        except Exception as e:
            raise ValueError(f"Error loading dataset from folder or hf id: {e}")
    raise ValueError(
        f"Failed to find a way to load the provided dataset path {training_data_path}"
    )


def load_model_file_from_hf(model_name_or_path: str, file_name: str) -> dict:
    """Load contens of a specific file of a model. Supports both the local file system and HF hub."""
    try:
        config_path = hf_hub_download(repo_id=model_name_or_path, filename=file_name)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    except Exception as e:
        return {}

    return config


def escape_newlines_in_strings(template_str: str) -> str:
    """Replace newlines in strings with \\n to aid in correctly rendering jinga template"""
    # This regex matches single or double quoted strings, non-greedy
    pattern = (
        r"""(['"])(.*?)(?<!\\)\1"""  # match quotes, content, and closing same quote
    )

    def replace_newlines(match):
        quote = match.group(1)
        content = match.group(2)
        # Replace real newlines with literal \n inside the string content
        content_escaped = content.replace("\n", "\\n")
        return f"{quote}{content_escaped}{quote}"

    # re.DOTALL makes '.' match newlines inside the string content
    return re.sub(pattern, replace_newlines, template_str, flags=re.DOTALL)


def get_model_path(model_name_or_path: str, unique_tag: str) -> str:
    """Given an indirect model name or path, pick out the exact model name"""
    model_name_or_path = Path(model_name_or_path)
    files_to_download = [
        "config.json",
        "tokenizer_config.json",
    ]

    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    else:
        BASE_DIR = Path(__file__).parent.parent
        cached_model_path = (
            BASE_DIR / "cached_files" / "models" / model_name_or_path / unique_tag
        )
        os.makedirs(cached_model_path, exist_ok=True)

        for filename in files_to_download:
            src = hf_hub_download(str(model_name_or_path), filename=filename)
            dst = os.path.join(cached_model_path, filename)
            shutil.copy(src, dst)
        model_name_or_path = cached_model_path

    return str(model_name_or_path)
