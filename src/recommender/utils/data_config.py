import os
import json
import yaml
from pathlib import Path
from recommender.utils.data_processing import (
    load_training_data,
    load_model_file_from_hf,
)
from recommender.utils.train_config import (
    fetch_from_knowledge_base,
    is_model_type_moe,
)
from pathlib import Path

import os
import json


def get_model_for_chat_template_mapping(model_name_or_path):
    # Direct
    model_name = model_name_or_path.split("/")[-2]

    base_dir = Path(__file__).parent.parent
    config_path = (
        base_dir / "knowledge_base" / "default_mappings" / "chat_template_map.yaml"
    )
    with open(config_path, "r") as file:
        MODEL_NAME_FOR_CHAT_TEMPLATE_MAPPING = yaml.safe_load(file)

    for model_id, keywords in MODEL_NAME_FOR_CHAT_TEMPLATE_MAPPING.items():
        for keyword in keywords:
            if keyword in model_name:
                return model_id

    is_moe_model = is_model_type_moe(model_name_or_path)
    if is_moe_model:
        return "ibm-granite/granite-4.0-tiny-preview"

    return None


def fetch_chat_template(model_name_or_path: str):
    """Given a model HF ID or Path, fetch the chat template (instruct model)"""
    if os.path.isdir(model_name_or_path):
        with open(
            f"{model_name_or_path}/tokenizer_config.json", "r", encoding="utf-8"
        ) as f:
            config = json.load(f)
            if "chat_template" not in config:
                model_name = "ibm-granite/" + model_name_or_path.split("/")[-2].replace(
                    "base", "instruct"
                )
                config = load_model_file_from_hf(model_name, "tokenizer_config.json")

                # If couldn't fetch instruct model from HF, fallback to defaults mapping
                model_name = get_model_for_chat_template_mapping(model_name_or_path)
                if model_name:
                    config = load_model_file_from_hf(
                        model_name, "tokenizer_config.json"
                    )

    elif (
        result := fetch_from_knowledge_base(
            model_name_or_path, config_folder="chat_template"
        )
    )[1]:
        config = result[0]
        additional_special_tokens = fetch_from_knowledge_base(
            model_name_or_path, config_folder="additional_special_tokens"
        )[0]
        config.update(additional_special_tokens) if additional_special_tokens else None
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

    for col in COMMON_INPUT_KEYS:
        if any(key in col.lower() for key in columns):
            input_col = col
            break

    for col in COMMON_RESPONSE_KEYS:
        if any(key in col.lower() for key in columns):
            output_col = col
            break

    return input_col, output_col


def has_any_key_containing(example, key_substrings):
    return any(
        any(sub in key.lower() for sub in key_substrings)
        for key in example.keys()
    )
