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

# NOTE: copied from older commit

def fuzzy_params(catalog: Dict[str, dict]) -> Dict[str, set]:
    """{tool_name: {tên tham số fuzzy}} — evaluate.py dùng để chấm token-F1."""
    res = {}
    for n, t in catalog.items():
        props = (t.get("parameters") or {}).get("properties") or {}
        res[n] = {p for p, s in props.items() if s.get("x-match") == "fuzzy"}
    return res

def compact_tool(tool: dict) -> dict:
    """Rút gọn 1 tool, gắn nhãn từng tham số: exact | fuzzy | ref | one_of."""
    params = tool.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    groups = params.get("x-one-of") or []
    in_group = {p: i for i, g in enumerate(groups) for p in g}
    out = []
    for name, spec in props.items():
        if "x-ref" in spec:
            label = "ref"
        elif spec.get("x-match") == "fuzzy":
            label = "fuzzy"
        elif name in in_group:
            label = "one_of"
        else:
            label = "exact"
        p = {"name": name, "type": spec.get("type"), "required": name in required,
             "label": label}
        if "enum" in spec:
            p["enum"] = spec["enum"]
        if "x-ref" in spec:
            p["x_ref"] = spec["x-ref"]
        if name in in_group:
            p["group"] = in_group[name]
        if spec.get("description"):
            p["description"] = spec["description"]
        out.append(p)
    return {"name": tool["name"], "app": tool.get("app"), "kind": tool.get("kind"),
            "description": tool.get("description", ""), "returns": tool.get("returns", ""),
            "params": out, "one_of": groups}

