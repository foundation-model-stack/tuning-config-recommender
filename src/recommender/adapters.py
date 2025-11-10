from recommender.actions import IR
from recommender.utils.data_processing import get_model_path
from recommender.rule_engine import RuleEngine
from copy import deepcopy


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
        model_name_or_path = get_model_path(model_name_or_path, unique_tag=unique_tag)
        train_config["model_name_or_path"] = model_name_or_path
        ir = IR(
            train_config=train_config,
            compute_config=compute_config,
            dist_config=dist_config,
            data_config=data_config,
        )
        ir_to_apply, json_patches = re.apply(ir=deepcopy(ir))
        return ir_to_apply, json_patches
