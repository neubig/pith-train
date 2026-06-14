"""Run a 2-node PithTrain smoke on ADP-derived text with Qwen3.5-35B-A3B."""

import os
from datetime import timedelta
from pathlib import Path

from pithtrain.modules.logging import LoggingWandbCfg
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch


cfg = PretrainLMCfg()

distributed = cfg.distributed
distributed.context_parallel_size = int(os.environ.get("PITHTRAIN_CONTEXT_PARALLEL_SIZE", "1"))
distributed.pipeline_parallel_size = int(os.environ.get("PITHTRAIN_PIPELINE_PARALLEL_SIZE", "2"))
distributed.expert_parallel_size = int(os.environ.get("PITHTRAIN_EXPERT_PARALLEL_SIZE", "8"))
distributed.timeout = timedelta(minutes=int(os.environ.get("PITHTRAIN_TIMEOUT_MINUTES", "60")))

training = cfg.training
training.model = Path(
    os.environ.get("PITHTRAIN_MODEL_CONFIG", "/project/flame/gneubig/adp/models/Qwen3.5-35B-A3B")
)
training.optimizer = "Adam"
training.scheduler = "CosineAnnealing"
training.max_lr = float(os.environ.get("PITHTRAIN_MAX_LR", "1.0e-5"))
training.min_lr = float(os.environ.get("PITHTRAIN_MIN_LR", "1.0e-6"))
training.warmup_steps = int(os.environ.get("PITHTRAIN_WARMUP_STEPS", "1"))
training.max_steps = int(os.environ.get("PITHTRAIN_MAX_STEPS", "12"))
training.micro_batch_size = int(os.environ.get("PITHTRAIN_MICRO_BATCH_SIZE", "1"))
training.global_batch_size = int(os.environ.get("PITHTRAIN_GLOBAL_BATCH_SIZE", "32"))
training.sequence_length = int(os.environ.get("PITHTRAIN_SEQUENCE_LENGTH", "2048"))
training.dataset = Path(
    os.environ.get(
        "PITHTRAIN_DATASET",
        "/home/gneubig/workspace/project/7511d8c2946a4a53a9b2a4f643840479/"
        "pithtrain_runs/adp_qwen35_35b_a3b_smoke/toktxt/qwen35",
    )
)
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = float(os.environ.get("PITHTRAIN_MOE_LOAD_BALANCE_COEF", "1.0e-3"))
training.fp8_training = os.environ.get("PITHTRAIN_FP8_TRAINING", "disabled")
training.save_interval = (
    int(os.environ["PITHTRAIN_SAVE_INTERVAL"]) if "PITHTRAIN_SAVE_INTERVAL" in os.environ else None
)

checkpoint_root = os.environ.get(
    "PITHTRAIN_SAVE_LOCATION",
    "/project/flame/gneubig/adp/pithtrain_checkpoints/qwen3.5-35b-a3b",
)
training.save_location = Path(checkpoint_root) if checkpoint_root else None

logging = cfg.logging
logging.wandb = LoggingWandbCfg()
logging.wandb.entity = os.environ.get("WANDB_ENTITY") or None
logging.wandb.project = os.environ.get("WANDB_PROJECT", "adp-experiments")
logging.wandb.name = os.environ.get(
    "WANDB_NAME",
    "adp-pithtrain-qwen35-35b-a3b-adp-seq2048-gbs32-pp2-ep8-2node-h100-smoke",
)
logging.wandb.group = os.environ.get("WANDB_GROUP", "adp-pithtrain-smoke")


if __name__ == "__main__":
    launch(cfg)
