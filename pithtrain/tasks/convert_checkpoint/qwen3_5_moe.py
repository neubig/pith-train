"""Qwen3.5 MoE checkpoint conversion helpers."""

import json
import re
from logging import Logger
from pathlib import Path
from typing import Dict

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open


class Qwen3_5MoeConverter:
    """Convert Qwen3.5 MoE HF checkpoints to PithTrain's text-only layout."""

    name = "qwen3_5_moe"

    def detect_hf(self, load_path: Path) -> bool:
        config_path = Path(load_path, "config.json")
        if not config_path.exists():
            return False
        with open(config_path) as f:
            return json.load(f).get("model_type") == "qwen3_5_moe"

    def detect_dcp(self, metadata) -> bool:
        return any(
            key.startswith("app.model.layers.") and ".linear_attn." in key
            for key in metadata.state_dict_metadata
        )

    def hf2dcp(self, load_path: Path, save_path: Path, stdout: Logger) -> None:
        with open(Path(load_path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]

        shard_files = set(weight_map.values())
        stdout.info(
            "Converting Qwen3.5 MoE text checkpoint from %s (%d shards)"
            % (load_path, len(shard_files))
        )

        model_state_dict: Dict[str, torch.Tensor] = dict()
        for i, shard_file in enumerate(sorted(shard_files), start=1):
            stdout.info("Reading shard %d/%d: %s" % (i, len(shard_files), shard_file))
            with safe_open(str(Path(load_path, shard_file)), framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if key == "lm_head.weight":
                        model_state_dict[key] = tensor
                        continue
                    if not key.startswith("model.language_model."):
                        continue

                    canonical = key.removeprefix("model.language_model.")
                    if canonical.endswith(".mlp.experts.gate_up_proj"):
                        prefix = canonical.removesuffix(".gate_up_proj")
                        gate, up = tensor.chunk(2, dim=1)
                        for idx in range(tensor.shape[0]):
                            expert_prefix = prefix.replace(".experts.", ".experts.%d." % idx, 1)
                            model_state_dict[expert_prefix + ".gate_proj.weight"] = gate[
                                idx
                            ].contiguous()
                            model_state_dict[expert_prefix + ".up_proj.weight"] = up[
                                idx
                            ].contiguous()
                    elif canonical.endswith(".mlp.experts.down_proj"):
                        prefix = canonical.removesuffix(".down_proj")
                        for idx in range(tensor.shape[0]):
                            expert_prefix = prefix.replace(".experts.", ".experts.%d." % idx, 1)
                            model_state_dict[expert_prefix + ".down_proj.weight"] = tensor[
                                idx
                            ].contiguous()
                    else:
                        model_state_dict[canonical] = tensor

        save_path.mkdir(parents=True, exist_ok=True)
        dcp.save({"app": {"model": model_state_dict}}, checkpoint_id=save_path, no_dist=True)
        stdout.info("Saved DCP checkpoint to %s (%d weights)" % (save_path, len(model_state_dict)))

    def postprocess_canonical(
        self, canonical: Dict[str, torch.Tensor], stdout: Logger
    ) -> Dict[str, torch.Tensor]:
        indexed = re.compile(r"(.*\.mlp\.experts)\.(\d+)\.(.*)")
        expert_tensors: Dict[str, Dict[int, torch.Tensor]] = {}
        plain: Dict[str, torch.Tensor] = {}

        for key, tensor in canonical.items():
            match = indexed.match(key)
            if match:
                prefix, idx_str, suffix = match.group(1), match.group(2), match.group(3)
                stacked_key = "%s.%s" % (prefix, suffix)
                expert_tensors.setdefault(stacked_key, {})[int(idx_str)] = tensor
            else:
                plain[key] = tensor

        result: Dict[str, torch.Tensor] = {}
        for key, tensor in plain.items():
            result[key if key == "lm_head.weight" else "language_model." + key] = tensor

        consumed = set()
        for key, by_idx in sorted(expert_tensors.items()):
            if key in consumed:
                continue
            if key.endswith(".gate_proj.weight"):
                prefix = key.removesuffix(".gate_proj.weight")
                up_key = prefix + ".up_proj.weight"
                if up_key not in expert_tensors:
                    raise KeyError("Missing paired Qwen3.5 expert tensor: %s" % up_key)
                gate = torch.stack([t for _, t in sorted(by_idx.items())])
                up = torch.stack([t for _, t in sorted(expert_tensors[up_key].items())])
                result["language_model." + prefix + ".gate_up_proj"] = torch.cat(
                    (gate, up), dim=1
                ).contiguous()
                consumed.add(key)
                consumed.add(up_key)
            elif key.endswith(".up_proj.weight"):
                continue
            elif key.endswith(".down_proj.weight"):
                stacked = torch.stack([t for _, t in sorted(by_idx.items())])
                result[
                    "language_model." + key.removesuffix(".weight")
                ] = stacked.contiguous()
                consumed.add(key)
            else:
                raise ValueError("Unexpected Qwen3.5 expert tensor key: %s" % key)

        stdout.info(
            "Postprocessed Qwen3.5 MoE canonical tensors: %d -> %d" % (len(canonical), len(result))
        )
        return result
