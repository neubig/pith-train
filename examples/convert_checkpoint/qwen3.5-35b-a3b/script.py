"""Convert the local Qwen3.5-35B-A3B HF checkpoint to PithTrain DCP."""

import os
from pathlib import Path

from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, launch


cfg = ConvertCheckpointCfg()
cfg.operation = "hf2dcp"
cfg.load_path = Path(
    os.environ.get("QWEN35_HF", "/project/flame/gneubig/adp/models/Qwen3.5-35B-A3B")
)
root = Path(
    os.environ.get(
        "PITHTRAIN_QWEN35_CHECKPOINT_ROOT",
        "/project/flame/gneubig/adp/pithtrain_checkpoints/qwen3.5-35b-a3b",
    )
)
cfg.save_path = root / "torch-dcp" / "step-00000000"


if __name__ == "__main__":
    launch(cfg)
