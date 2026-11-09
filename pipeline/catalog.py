"""Nạp catalog, lọc tool theo installed_apps, rút gọn schema cho prompt LLM.

Catalog: {"apps": [{"id", "tools": [{"name", "kind", "description", "parameters",
"returns", "app"}]}]}. installed_apps của sample: {app_id: {"tools": [tên tool]}}.
Chỉ đọc <split>_tools_catalog.json (được phép), KHÔNG đọc state.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

ASK = "sys.ask_missing_info"


def load_catalog(path) -> Dict[str, dict]:
    """Trả {tool_name: tool}, giữ thứ tự trong file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, dict] = {}
    for app in data.get("apps", []):
        for t in app.get("tools", []):
            t = dict(t)
            t.setdefault("app", app.get("id"))
            out[t["name"]] = t
    return out


def filter_tools(catalog: Dict[str, dict], installed_apps) -> Tuple[List[dict], List[str]]:
    """Trả (available_tools, tên_không_có_trong_catalog). Thứ tự ổn định (tất định)."""
    names: List[str] = []
    for app in (installed_apps or {}).values():
        for n in (app or {}).get("tools", []) or []:
            n = n["name"] if isinstance(n, dict) else n
            if n not in names:
                names.append(n)
    tools, missing = [], []
    for n in names:
        if n in catalog:
            tools.append(catalog[n])
        else:
            missing.append(n)
    if ASK in catalog and ASK not in names:
        tools.append(catalog[ASK])
    return tools, missing


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


def fuzzy_params(catalog: Dict[str, dict]) -> Dict[str, set]:
    """{tool_name: {tên tham số fuzzy}} — evaluate.py dùng để chấm token-F1."""
    res = {}
    for n, t in catalog.items():
        props = (t.get("parameters") or {}).get("properties") or {}
        res[n] = {p for p, s in props.items() if s.get("x-match") == "fuzzy"}
    return res


def resolve_layout(data_dir, split) -> Path:
    """Trả root (thư mục CHA chứa <split>/). Hỗ trợ cả layout Colab root/root.name/<split>."""
    root = Path(data_dir)
    if (root / split / f"{split}.jsonl").exists():
        return root
    if (root / root.name / split / f"{split}.jsonl").exists():
        return root / root.name
    raise FileNotFoundError(f"Không thấy {split}/{split}.jsonl trong {root}")
