# Tuning Config Recommender

⚡️ Supercharge your fine-tuning users with completely automated tuning configurations for their choice of model, and dataset enabling instant one-click tuning deployment 🚀.

## Installation

```
pip install -e .
```

or from PyPI

```
pip install tuning_config_recommender
```

## Library Usage

An example is given in `lib_usage.py` file which generates tuning configuration for [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning) stack given model `ibm-granite/granite-4.0-h-350m` and dataset with HF ID `ought/raft`.

```
python lib_usage.py
```

## CLI usage

```
python src/recommender/cli.py --tuning-data-config ./artifacts/test/data_config.yaml --accelerate-config ./artifacts/test/accelerate_config.yaml --tuning-config ./artifacts/test/tuning_config.yaml --compute-config ./artifacts/test/compute_config.yaml --output-dir ./output
```

Custom rules-dir usage

```
python src/recommender/cli.py --tuning-data-config ./artifacts/test/data_config.yaml --accelerate-config ./artifacts/test/accelerate_config.yaml --tuning-config ./artifacts/test/tuning_config.yaml --compute-config ./artifacts/test/compute_config.yaml --output-dir ./output --rules-dir custom_rules_dir
```

Writing custom action rules for custom modification would require following the below
1. Should start with name "Custom_"
2. Should subclass from `Action` class

An example can be found at [custom_rules_dir](./custom_rules_dir/).

## API Usage

After installing it as a module you can start an API as

```
uvicorn tuning_config_recommender.api:app --reload
```

`/docs` endpoint provides details on the endpoint to make requests.

## Architecture

![](./artifacts/architecture.png)

### Concepts

#### Intermediate Representation (IR)
IR is the standard format that all input formats have to be converted to so that rule engine can work on the IR without worrying about the exact input or output formats which could differ from one stack to another. There is a caveat with IR, not all arguments in the IR are strictly defined in the sense that the actions currently written may add some new arguments and IR or rule engine are restrictive about it. Its up to do the adapter if it wishes to consume it or not for the target format.

#### Action
An action takes IR as input at its current state and performs some heuristics and constructs a new IR object which is used as a JSON Merge patch by the rule-engine. Addtionally, the returned new IR object can also hold various information about the patch such as severity, type and natural language comments. As shown in the architecture, an action would be called multiple times by the rule engine until it explicitly calls out skip. When to skip is the responsibility of the action which could be a heuristic based on the state of the IR when its called. Some example actions can be seen [here](./src/recommender/actions).

#### Rule Engine
Rule engine passes the IR across actions in the sequence they are defined and collects all JSON merge patches. These JSON merge patches are then applied over the IR. This process is again iterated until all actions call out for a skip. Finally, JSON patches (is different from the merge patch) with respect to the orginal IR provided to the rule engine is prepared while preserving all the metadata (comments etc) for each of the patch along with the final IR to adapters.

#### Adapter
Adapter converts source format to required IR format and consumes final IR and json patches as needed to deliver the target format. Adapters can be found [here](./src/recommender/adapters.py).
