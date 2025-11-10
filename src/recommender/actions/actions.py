from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import StrEnum, auto
import jsonpatch
from loguru import logger


class PatchLevel(StrEnum):
    MANDATORY = auto()
    SUGGESTION = auto()
    USER_INTERVENTION = auto()


class PatchType(StrEnum):
    SYSTEM_PERFORMANCE = auto()
    MODEL_QUALITY = auto()
    COMPATIBILITY = auto()


class Comment:
    comment: str = ""

    def __init__(self, cmt: str = ""):
        self.comment = cmt

    def add(self, cmt: str):
        self.comment = self.comment + "\n" + cmt

    def __repr__(self):
        return self.comment


@dataclass
class IR:
    train_config: Optional[Dict] = field(default_factory=dict)
    compute_config: Optional[Dict] = field(default_factory=dict)
    dist_config: Optional[Dict] = field(default_factory=dict)
    data_config: Optional[Dict] = field(default_factory=dict)
    # For json merge patch metadata
    level: Any = None
    type: Any = None
    effect: List = None
    comment: str = None

    def __post_init__(self):
        if not self.effect:
            self.effect = self.type

    def update(self, json_merge_patch):
        data_keys = [
            "train_config",
            "compute_config",
            "dist_config",
            "data_config",
        ]
        for key in data_keys:
            if key in json_merge_patch.__dict__ and json_merge_patch.__dict__[key]:
                self.__dict__[key].update(json_merge_patch.__dict__[key])

    def to_dict(self):
        return self.__dict__

    def get_json_patch(self, ir):
        patch = list(jsonpatch.JsonPatch.from_diff(self.__dict__, ir.__dict__))
        logger.debug(f"#######\nJSON patch {patch}\nJSON merge patch\n {ir}\n#######")
        if not patch:
            patch = []
        return patch


class Action:
    skip: bool = False
    depends_on_train_config: bool = False
    depends_on_compute_config: bool = False
    depends_on_dist_config: bool = False
    depends_on_data_preprocessor: bool = False
    depends_on_dataset: bool = False
    json_merge_patches: List[IR] = []
    json_patches_and_comment_wrt_source: List[Dict] = []

    def heuristic_skip(self, ir: IR) -> bool:
        """Given the existing input, this function does some heuristic analysis
        to either skip and keep the existing config as is or not skip and apply the action.

        Args:
            ir (IR): intermediate representation object

        Returns:
            bool: to either skip or not
        """
        return False

    def apply(self, ir: IR):
        pass
