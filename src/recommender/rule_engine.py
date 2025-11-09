from typing import List, Dict
from recommender.actions import Action, IR, ACTIONS
import os
from copy import deepcopy
from loguru import logger
from tqdm import tqdm
from recommender.utils import set_difference, set_issubset


class RuleEngine:
    actions: List[Action] = []
    ir_pipeline: List[IR] = []

    def __init__(self):
        pass

    def _validate_action(self, action: Action):
        expected_arg_count = 2
        if action.apply.__code__.co_argcount != expected_arg_count:
            raise ValueError(
                f"action {action.__class__.__name__} should have {expected_arg_count}, but got {action.apply.__code__.co_argcount}"
            )
        logger.debug(f"action {action.__class__.__name__} is valid!")

    def register_action(self, action: Action):
        self._validate_action(action=action)
        self.actions.append(action)
        logger.debug(f"action {action.__class__.__name__} registered!")

    def register_all_inbuilt_actions(self):
        for action_cls in ACTIONS:
            self.register_action(action_cls())
        logger.debug(f"All actions registered!")

    def _get_json_patch_from_merge_patch(
        self, json_merge_patch: IR, source_ir: IR, ir_to_patch: IR
    ):
        ir_to_patch.update(json_merge_patch=json_merge_patch)
        return source_ir.get_json_patch(ir_to_patch)

    def run_all_actions(self, ir: IR):
        running_ir = ir
        for action in tqdm(
            self.actions, total=(len(self.actions)), desc="Iterating over actions"
        ):
            json_merge_patch: IR = action.apply(deepcopy(running_ir))
            if not json_merge_patch:
                continue
            json_patch = self._get_json_patch_from_merge_patch(
                json_merge_patch, deepcopy(self.ir_pipeline[0]), deepcopy(running_ir)
            )
            logger.debug(
                f"action {action.__class__.__name__} applied, returned json merge patch {json_merge_patch} and json patch {json_patch}"
            )
            action.json_patches_and_comment_wrt_source.append(
                {
                    "comment": json_merge_patch.comment,
                    "json_patch": json_patch,
                    "json_merge_patch": json_merge_patch,
                    "stage_source_ir": deepcopy(running_ir),
                }
            )
            running_ir.update(json_merge_patch)
        return running_ir

    def validate_and_maybe_fix_ir(self, ir: IR):
        if not os.path.exists(ir.train_config.get("model_name_or_path", None)):
            raise ValueError(
                f"Given model_name_or_path {ir.train_config.get("model_name_or_path", "")}"
                "is not accessible to rule engine"
            )
        if ir.train_config.get("tuning_strategy", None) not in [
            "lora",
            "full",
            "none",
            "alora",
        ]:
            raise ValueError(
                f"Tuning strategy {ir.train_config.get("tuning_strategy", "")} is not clear."
                "Should be one of lora, full, none"
            )
        if (
            ir.train_config.get("training_data_path", None)
            and len(ir.data_preprocessor.get("datasets", [])) > 0
        ):
            ir.train_config.pop("training_data_path")
        logger.debug(f"IR {ir} is valid!")
        return ir

    def apply(self, ir: IR):
        max_iterations = 20
        ir_to_apply: IR = deepcopy(ir)
        self.ir_pipeline.append(deepcopy(ir))
        while any([not action.skip for action in self.actions]) and max_iterations:
            ir_to_apply = self.validate_and_maybe_fix_ir(ir_to_apply)
            ir_to_apply = self.run_all_actions(ir_to_apply)
            self.ir_pipeline.append(deepcopy(ir_to_apply))
            max_iterations -= 1
        # extracting comments for json patches
        json_patches = self.ir_pipeline[0].get_json_patch(ir_to_apply)
        final_json_patches_with_comment: List[Dict] = []
        _json_patches_that_have_comments = []
        for action in self.actions:
            if len(action.json_patches_and_comment_wrt_source):
                if set_issubset(
                    json_patches,
                    action.json_patches_and_comment_wrt_source[-1]["json_patch"],
                ):
                    final_json_patches_with_comment.append(
                        action.json_patches_and_comment_wrt_source[-1]
                    )
                    _json_patches_that_have_comments.extend(
                        action.json_patches_and_comment_wrt_source[-1]["json_patch"]
                    )
        final_json_patches_with_comment.append(
            {
                "comment": "",
                "json_patch": set_difference(
                    json_patches, _json_patches_that_have_comments
                ),
            }
        )
        return ir_to_apply, final_json_patches_with_comment
