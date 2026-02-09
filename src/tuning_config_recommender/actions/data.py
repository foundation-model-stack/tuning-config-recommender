from tuning_config_recommender.utils.data_config import (
    determine_input_and_response_text,
    fetch_chat_template,
    has_any_key_containing,
)
from tuning_config_recommender.utils.data_processing import (
    escape_newlines_in_strings,
    load_training_data,
)

from .actions import IR, Action, Comment, PatchLevel, PatchType


class ApplyDataFormat(Action):
    def _is_data_in_required_format(self, dataset_path: str) -> bool:
        raise NotImplementedError(
            "Data format validation should be implemented by child data based action class."
        )

    def _is_data_tokenized(self, path):
        data = load_training_data(path)
        if not data or not isinstance(data, list):
            return False
        if not isinstance(data[0], dict):
            return False
        tokenized_fields = {"input_ids", "labels", "attention_mask"}
        return any(field in data[0] for field in tokenized_fields)

    def heuristic_skip(self, ir):
        if ir.tuning_config.get(
            "training_data_path", None
        ) or ir.tuning_data_config.get("datasets", None):
            if ir.tuning_data_config.get("datasets", None):
                for dataset in ir.tuning_data_config["datasets"]:
                    if not len(dataset.get("data_paths", [])):
                        # TODO: instead of skipping
                        # we should return IR with type USER_INTERVENTION
                        return True
                    for path in dataset.get("data_paths"):
                        if self._is_data_in_required_format(
                            path
                        ) and not self._is_data_tokenized(path):
                            # NOTE: we check for one path to be in format
                            # while all paths are checked in action.apply
                            # implementation
                            return False
            if ir.tuning_config.get("training_data_path", None):
                return not self._is_data_in_required_format(
                    ir.tuning_config.get("training_data_path", None)
                )
        return True

    def _are_all_datapaths_in_format(self, data_paths):
        return not any(
            [not self._is_data_in_required_format(path) for path in data_paths]
        )


class ApplyQAFormat(ApplyDataFormat):
    def _is_data_in_required_format(self, dataset_path: str) -> bool:
        data = load_training_data(dataset_path)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        COMMON_INPUT_KEYS = [
            "input",
            "instruction",
            "prompt",
            "question",
            "tweet_text",
            "query",
            "source",
            "tweet text",
        ]
        COMMON_RESPONSE_KEYS = [
            "output",
            "response",
            "answer",
            "label",
            "text_label",
            "target",
            "completion",
        ]

        if has_any_key_containing(data, COMMON_INPUT_KEYS) and has_any_key_containing(
            data, COMMON_RESPONSE_KEYS
        ):
            return True
        return False

    def _is_dataset_in_required_format(self, dataset: dict) -> bool:
        # TODO: This can turn out to be an time-intensive operation
        # we should think if we want to do this.
        # ideally should iterate over each datapath using _is_data_in_required_format
        return True

    def _get_values_for_given_datapath(self, dataset_path: str):
        # TODO: Actions should made aware of the existing user changes
        # right now they work in replace-everything-first approach
        input_text, response_text = determine_input_and_response_text(dataset_path)
        template = f'"### Input: {{{{ {input_text} }}}}\\n\\n### Response: {{{{ {response_text} }}}}"'
        formatted_text_column_name = "formatted_qa_data"
        dataset_text_field = "formatted_qa_data"
        response_template = "\\n### Response:"
        return {
            "template": template,
            "input_text": input_text,
            "response_text": response_text,
            "formatted_text_column_name": formatted_text_column_name,
            "dataset_text_field": dataset_text_field,
            "response_template": response_template,
            "data_handlers": [
                {
                    "name": "apply_custom_jinja_template",
                    "arguments": {
                        "remove_columns": "all",
                        "batched": False,
                        "fn_kwargs": {
                            "formatted_text_column_name": formatted_text_column_name,
                            "template": template,
                        },
                    },
                }
            ],
        }

    def _get_values_for_given_dataset(self, dataset: dict):
        return self._get_values_for_given_datapath(dataset.get("data_paths")[0])

    def apply(self, ir: IR, actions_meta: list[str]) -> IR:
        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return

        # TODO: Actions should made aware of the existing user changes
        # right now they work in replace-everything-first approach

        if ir.tuning_config.get("training_data_path", None):
            ir.tuning_data_config["dataprocessor"] = {
                "type": "default",
                "streaming": False,
            }
            ir.tuning_data_config["datasets"] = [
                {
                    "name": "dataset_from_inputs",
                    "data_paths": [ir.tuning_config.get("training_data_path")],
                    "data_handlers": {},
                }
            ]

        for dataset in ir.tuning_data_config["datasets"]:
            if not self._is_data_in_required_format(dataset["data_paths"][0]):
                continue
            values_to_set = self._get_values_for_given_dataset(dataset)
            ir.tuning_config.update(
                {
                    "dataset_text_field": values_to_set["dataset_text_field"],
                    "response_template": values_to_set["response_template"],
                }
            )
            dataset["data_handlers"] = values_to_set["data_handlers"]
        ir.comment = Comment(
            "This data config is used for formatting QA datasets for training"
        )
        ir.type = PatchType.COMPATIBILITY
        ir.level = PatchLevel.MANDATORY
        self.json_merge_patches.append(ir)
        self.skip = True
        return ir


class ApplyChatFormat(ApplyDataFormat):
    CHAT_STYLE_KEYS = ["messages", "conversations", "dialogues", "chat", "turns"]

    def _is_data_in_required_format(self, dataset_path: str) -> bool:
        data = load_training_data(dataset_path)
        columns = data[0].keys()
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        if has_any_key_containing(data, self.CHAT_STYLE_KEYS):
            for chat_key in self.CHAT_STYLE_KEYS:
                if chat_key in columns:
                    val = data[chat_key]
                    if isinstance(val, list):
                        if all(
                            isinstance(m, dict) and "role" in m and "content" in m
                            for m in val
                        ):
                            return True
        return False

    def _is_dataset_in_required_format(self, dataset: dict) -> bool:
        # TODO: This can turn out to be an time-intensive operation
        # we should think if we want to do this.
        # ideally should iterate over each datapath using _is_data_in_required_format
        return True

    def _get_values_for_given_datapath(
        self, dataset_path: str, model_name_or_path: str, max_seq_length: int
    ):
        # TODO: Actions should made aware of the existing user changes
        # right now they work in replace-everything-first approach

        # TODO: add more checks for specials tokens present or not
        chat_template, special_tokens = fetch_chat_template(model_name_or_path)
        if chat_template:
            chat_template = escape_newlines_in_strings(chat_template)
            chat_template = "{% raw %}\n  " + chat_template + "\n  {% endraw %}"
        data = load_training_data(dataset_path)
        columns = data[0].keys()
        for chat_key in self.CHAT_STYLE_KEYS:
            if chat_key in columns:
                conversation_column_name = chat_key
                break

        return {
            "chat_template": chat_template,
            "conversation_column_name": conversation_column_name,
            "data_handlers": [
                {
                    "name": "tokenize_and_apply_chat_template_with_masking",
                    "arguments": {
                        "remove_columns": "all",
                        "fn_kwargs": {
                            "max_seq_length": max_seq_length,
                            "conversation_column_name": conversation_column_name,
                        },
                    },
                }
            ],
        }

    def _get_values_for_given_dataset(
        self, dataset: dict, model_name_or_path: str, max_seq_length: int
    ):
        return self._get_values_for_given_datapath(
            dataset.get("data_paths")[0], model_name_or_path, max_seq_length
        )

    def apply(self, ir: IR, actions_meta: list[str]) -> IR:
        if self.heuristic_skip(ir) or self.skip:
            self.skip = True
            return

        # TODO: Actions should made aware of the existing user changes
        # right now they work in replace-everything-first approach

        if ir.tuning_config.get("training_data_path", None):
            ir.tuning_data_config["dataprocessor"] = {
                "type": "default",
                "streaming": False,
            }
            ir.tuning_data_config["datasets"] = [
                {
                    "name": "dataset_from_inputs",
                    "data_paths": [ir.tuning_config.get("training_data_path")],
                    "data_handlers": {},
                }
            ]
        for dataset in ir.tuning_data_config["datasets"]:
            if not self._is_data_in_required_format(dataset["data_paths"][0]):
                continue
            values_to_set = self._get_values_for_given_dataset(
                dataset,
                ir.tuning_config["model_name_or_path"],
                ir.tuning_config.get("max_seq_length", 4096),
            )
            dataset["data_handlers"] = values_to_set["data_handlers"]
            # TODO: all datasets can only use one chat template
            # so we should check if we find a different chat template
            # return it as user_intervention
            ir.tuning_data_config["chat_template"] = values_to_set["chat_template"]
        ir.comment = Comment(
            "This data config is used for formatting chat datasets for training"
        )
        ir.type = PatchType.COMPATIBILITY
        ir.level = PatchLevel.MANDATORY
        self.json_merge_patches.append(ir)
        self.skip = True
        return ir
