import json
import os
from pathlib import Path

import yaml

from tuning_config_recommender.utils.data_processing import (
    load_model_file_from_hf,
    load_training_data,
)
from tuning_config_recommender.utils.train_config import (
    fetch_from_knowledge_base,
)


def fetch_chat_template(model_name_or_path: str):
    """Given a model HF ID or Path, fetch the chat template (instruct model)"""
    if os.path.isdir(model_name_or_path):
        with open(f"{model_name_or_path}/tokenizer_config.json", encoding="utf-8") as f:
            config = json.load(f)

    elif (
        result := fetch_from_knowledge_base(
            model_name_or_path, kb_section="chat_template"
        )
    )[1]:
        config = {"chat_template": result[0]}
        additional_special_tokens, _ = fetch_from_knowledge_base(
            model_name_or_path, kb_section="additional_special_tokens"
        )
        if additional_special_tokens:
            config["additional_special_tokens"] = additional_special_tokens

    else:
        try:
            config = load_model_file_from_hf(
                model_name_or_path, "tokenizer_config.json"
            )

        except Exception as e:
            return f"Error: {e}", None

    chat_template = config.get("chat_template", None)
    additional_special_tokens = config.get("additional_special_tokens", None)

    return chat_template, additional_special_tokens


def determine_input_and_response_text(training_data_path: str) -> dict:
    """Determine the input and response field for the data formating template (Q/A format dataset)"""
    data = load_training_data(training_data_path)
    data_item = data[0]
    columns = list(data_item.keys())
    columns = [k.lower() for k in columns]

    COMMON_INPUT_KEYS = [
        "tweet_text",
        "tweet text",
        "instruction",
        "prompt",
        "question",
        "input",
        "query",
        "source",
        "content",
    ]

    COMMON_RESPONSE_KEYS = [
        "text_label",
        "label",
        "response",
        "answer",
        "output",
        "target",
        "completion",
    ]

    input_col = "input"
    output_col = "output"

    for col in columns:
        if any(key in col.lower() for key in COMMON_INPUT_KEYS):
            input_col = col
            break

    for col in columns:
        if any(key in col.lower() for key in COMMON_RESPONSE_KEYS):
            output_col = col
            break

    return input_col, output_col


def has_any_key_containing(example, key_substrings):
    return any(
        any(sub in key.lower() for sub in key_substrings) for key in example.keys()
    )
