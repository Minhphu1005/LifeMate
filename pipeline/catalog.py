import json
from pathlib import Path

def load_catalog(path):
    path = Path(path);
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f);
    catalog: Dict[str, dict] = {};
    for app in data.get("apps", []):
        for tool in app.get("tools", []):
            catalog[tool["name"]] = tool;
    return catalog;

def resolve_layout(data_dir, split):
    d = Path(data_dir).resolve();
    if (d / split / f"{split}.jsonl").exists():
        return d;
    if (d / d.name / split / f"{split}.jsonl").exists():
        return d / d.name;
    if (d / f"{split}.jsonl").exists():
        return d.parent;
    raise FileNotFoundError();

