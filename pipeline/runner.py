from .stubs import DummyLLM, ReadProbeLLM;
from .catalog import *;

def run(cfg: dict, split: str, data_dir, out, limit: int = 0):
    print(split, data_dir, out);

    root = resolve_layout(data_dir, split);
    print(f"Found {split}.jsonl at ", root);

    pass;

