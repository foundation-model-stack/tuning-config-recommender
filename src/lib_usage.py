import importlib
import sys

from tuning_config_recommender.adapters import FMSAdapter, VanillaAdapter

user_folder = sys.argv[1]
sys.path.insert(0, user_folder)

module = importlib.import_module("a")  # a.py
MyClass = module.MyClass

if __name__ == "__main__":

    print("\n### Vanilla Adapter ###\n")
    adapter = VanillaAdapter()
    train_config = {
        "model_name_or_path": "ibm-granite/granite-3.1-8b-base",
        "training_data_path": "tatsu-lab/alpaca",
        "tuning_strategy": "full",
    }
    ir_to_apply, json_patches = adapter.execute(
        train_config,
        compute_config={},
        dist_config={},
        data_config={},
        unique_tag="gpq12df",
    )
    print(ir_to_apply, json_patches)

    print("\n### FMS Adapter ###\n")
    fms_adapter = FMSAdapter(base_dir="tmp/fms_full")

    result = fms_adapter.execute(
        train_config={
            "model_name_or_path": "ibm-granite/granite-3.1-8b-base",
            "training_data_path": "tatsu-lab/alpaca",
            "tuning_strategy": "full",
        },
        compute_config={},
        dist_config={},
        data_config={},
        unique_tag="gpq12df-fms",
        paths={},
    )
    print(result["launch_command"])
