"""Tokenize a small ADP-derived text corpus with the Qwen3.5 tokenizer."""

import os
from pathlib import Path

from pithtrain.tasks.tokenize_corpus import TokenizeCorpusCfg, launch


if __name__ == "__main__":
    cfg = TokenizeCorpusCfg()
    cfg.tokenizer_name = os.environ.get(
        "PITHTRAIN_TOKENIZER", "/project/flame/gneubig/adp/models/Qwen3.5-35B-A3B"
    )
    cfg.source_path = Path(
        os.environ.get(
            "PITHTRAIN_SOURCE_TEXT",
            "/home/gneubig/workspace/project/7511d8c2946a4a53a9b2a4f643840479/"
            "pithtrain_runs/adp_qwen3_30b_a3b_smoke/rawtxt",
        )
    )
    cfg.output_path = Path(
        os.environ.get(
            "PITHTRAIN_DATASET",
            "/home/gneubig/workspace/project/7511d8c2946a4a53a9b2a4f643840479/"
            "pithtrain_runs/adp_qwen35_35b_a3b_smoke/toktxt/qwen35",
        )
    )
    cfg.num_workers = int(os.environ.get("PITHTRAIN_TOKENIZE_WORKERS", "32"))
    launch(cfg)
