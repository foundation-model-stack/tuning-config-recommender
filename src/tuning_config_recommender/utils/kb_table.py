import fnmatch
import yaml
from pathlib import Path

_KB = None
_KB_TABLE = None


def _load_kb_yaml():
    """
    Load the knowledge base YAML once.
    """
    global _KB
    if _KB is not None:
        return _KB

    recommender_root = Path(__file__).resolve().parents[1]
    kb_path = recommender_root / "knowledge_base" / "knowledge_base.yaml"

    if not kb_path.exists():
        raise FileNotFoundError(f"KB not found: {kb_path}")

    with open(kb_path, "r", encoding="utf-8") as f:
        _KB = yaml.safe_load(f) or {}

    return _KB


def _build_kb_table():
    """
    Convert KB YAML into a flat table:
    [
        {
            model_pattern: str,
            section: str,
            payload: dict | str,
            priority: int
        }
    ]
    """
    global _KB_TABLE
    if _KB_TABLE is not None:
        return _KB_TABLE

    kb = _load_kb_yaml()
    table = []

    for section, payload in kb.get("general_defaults", {}).items():
        table.append(
            {
                "model_pattern": "*",
                "section": section,
                "payload": payload,
                "priority": 1000,
            }
        )

    models = kb.get("models", {})
    for idx, (model_name, model_cfg) in enumerate(models.items()):
        for section, payload in model_cfg.items():
            table.append(
                {
                    "model_pattern": model_name,
                    "section": section,
                    "payload": payload,
                    "priority": idx,
                }
            )

    table.sort(key=lambda r: r["priority"])
    _KB_TABLE = table
    return table


def query_kb(model_name: str, section: str):
    """
    Query KB table.
    Returns:
        (payload, found)
    """
    table = _build_kb_table()

    for row in table:
        if row["section"] != section:
            continue
        if fnmatch.fnmatch(model_name, row["model_pattern"]):
            return row["payload"], row["model_pattern"] != "*"

    return {}, False

