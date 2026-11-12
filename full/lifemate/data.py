"""Tìm thư mục split, nạp zaic_tools của split, đọc jsonl/catalog."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional, Tuple


def _has_split(d: Path, name: str) -> bool:
    return (d / f"{name}.jsonl").is_file() and (d / f"{name}_tools_catalog.json").is_file()


def locate_split(data_dir: str, split: Optional[str] = None) -> Tuple[str, Path, Path]:
    """Trả về (split, split_dir, root) sao cho ToolEnv(split, id, root=root) hoạt động.

    Chấp nhận: /data/<split>/..., hoặc /data trực tiếp chứa <split>.jsonl (BTC mount thư mục split vào /data).
    """
    d = Path(data_dir).resolve()
    if (d / d.name).is_dir() and not any((d / s).exists() for s in ("train", "public_test", "private_test")) \
            and not any(d.glob("*.jsonl")):
        d = d / d.name  # thư mục giải nén bị lồng: train+public_test/train+public_test/
    names = [split] if split else ["private_test", "public_test", "train"]
    # 1) /data/<split>/
    for n in names:
        if _has_split(d / n, n):
            return n, d / n, d
    # 2) /data chứa trực tiếp file của split
    for n in names:
        if _has_split(d, n):
            return n, d, _root_for(d, n)
    # 3) tự dò: bất kỳ thư mục con nào có <x>.jsonl + <x>_tools_catalog.json
    for cand in [d] + [p for p in sorted(d.iterdir()) if p.is_dir()]:
        for f in sorted(cand.glob("*.jsonl")):
            n = f.stem
            if n != "sample_submission" and _has_split(cand, n):
                return n, cand, _root_for(cand, n)
    raise FileNotFoundError(f"Không tìm thấy split trong {d}")


def _root_for(d: Path, n: str) -> Path:
    """Thư mục root sao cho root/<n> là d (ToolEnv cần root/<split>/<split>_environment/...)."""
    if d.name == n:
        return d.parent
    root = Path(tempfile.mkdtemp(prefix="zaic_root_"))
    os.symlink(d, root / n, target_is_directory=True)  # Linux/Docker; Windows cần quyền
    return root


def import_toolenv(split_dir: Path):
    """Nạp zaic_tools đúng phiên bản của split (nằm trong <split>/zaic_tools)."""
    p = str(split_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    from zaic_tools import ToolEnv  # type: ignore
    return ToolEnv


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_catalog_raw(split_dir: Path, split: str) -> dict:
    with open(split_dir / f"{split}_tools_catalog.json", encoding="utf-8") as f:
        return json.load(f)
