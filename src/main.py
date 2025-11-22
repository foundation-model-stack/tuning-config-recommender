from recommender.adapters import VanillaAdapter, FMSAdapter

if __name__ == "__main__":
    print("\n### Vanilla Adapter ###\n")
    adapter = VanillaAdapter()
    train_config = {
        "model_name_or_path": "ibm-granite/granite-4.0-h-350m",
        "training_data_path": "ought/raft",
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

    result = fms_adapter.run(
        train_config={
            "model_name_or_path": "ibm-granite/granite-4.0-h-350m",
            "training_data_path": "ought/raft",
            "tuning_strategy": "full",
        },
        unique_tag="gpq12df-fms",
    )
    print(result["launch_command"])

