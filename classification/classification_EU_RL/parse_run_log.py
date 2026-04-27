#!/usr/bin/env python3
"""Parse a training run log into evaluation_result.json and training_log.json.

This script is designed for logs produced by the classification/EU_RL training
scripts in this repository. It extracts:
- per-epoch metrics into training_log.json
- final evaluation metrics into evaluation_result.json

By default, output files are written next to the input run.log.
It also writes train_log.json as a compatibility alias for training_log.json.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

RE_SEED = re.compile(r"^setup random seed = (?P<seed>\d+)$")
RE_FORGET_SET_INFO = re.compile(
    r"^\[SEED (?P<seed>\d+)\] Forget set size: (?P<size>\d+), class dist: (?P<dist>\[.*\])$"
)
RE_RETAIN_FORGET_COUNTS = re.compile(r"^number of (?P<kind>retain|forget) dataset (?P<count>\d+)$")
RE_EPOCH_START = re.compile(r"^Epoch #(?P<epoch>\d+), Learning rate: (?P<lr>[-+\deE.]+)$")
RE_TRAIN_BATCH = re.compile(
    r"^Epoch:\s+\[(?P<epoch>\d+)\]\[(?P<batch>\d+)/(?P<total>\d+)\]\s+"
    r"Loss\s+(?P<loss_cur>[-+\deE.]+)\s+\((?P<loss_avg>[-+\deE.]+)\)\s+"
    r"Accuracy\s+(?P<acc_cur>[-+\deE.]+)\s+\((?P<acc_avg>[-+\deE.]+)\)\s+Time\s+(?P<time>[-+\deE.]+)$"
)
RE_VALID_BATCH = re.compile(
    r"^(?P<split>retain|forget|test):\s+\[(?P<batch>\d+)/(?P<total>\d+)\]\s+"
    r"Loss\s+(?P<loss_cur>[-+\deE.]+)\s+\((?P<loss_avg>[-+\deE.]+)\)\s+"
    r"Accuracy\s+(?P<acc_cur>[-+\deE.]+)\s+\((?P<acc_avg>[-+\deE.]+)\)$"
)
RE_VALID_FINAL_RAW_ACC = re.compile(r"^(?P<split>retain|forget|test)\s+acc:\s+(?P<acc>[-+\deE.]+)$")
RE_VALID_RAW_ACC = re.compile(r"^(?P<split>retain|forget|test)\s+Accuracy\s+(?P<acc>[-+\deE.]+)$")
RE_EPOCH_SUMMARY = re.compile(
    r"^Epoch\s+(?P<epoch>\d+)\s+\|\s+Retain Acc\s+(?P<retain_acc>[-+\deE.]+)\s+\|\s+"
    r"Forget Acc \(UA\)\s+(?P<forget_acc>[-+\deE.]+)\s+\|\s+Test Acc\s+(?P<test_acc>[-+\deE.]+)\s+\|\s+"
    r"Forget Loss\s+(?P<forget_loss>[-+\deE.]+)\s+\|\s+Retain Loss\s+(?P<retain_loss>[-+\deE.]+)\s+\|\s+"
    r"Param Update Norm\s+(?P<param_update_norm>[-+\deE.]+)$"
)
RE_TRAIN_ACC = re.compile(r"^Train Acc:(?P<train_acc>[-+\deE.]+)$")
RE_DURATION = re.compile(r"^one epoch duration:(?P<duration_sec>[-+\deE.]+)$")
RE_RL_REWARD = re.compile(
    r"^\s*RL Reward \| mean (?P<mean>[-+\deE.]+) std (?P<std>[-+\deE.]+) min (?P<min>[-+\deE.]+) max (?P<max>[-+\deE.]+)$"
)
RE_RL_OVERLAP = re.compile(
    r"^\[SEED (?P<seed>\d+) \| Epoch (?P<epoch>\d+)\] RL label overlap:\s+"
    r"(?P<overlap>[-+\deE.]+) .* ratio=(?P<ratio>[-+\deE.]+)x\)$"
)
RE_FORGET_TARGET_INTEGRITY = re.compile(
    r"^\[SEED (?P<seed>\d+)\] Forget targets at eval:\s+"
    r"(?P<negative_count>\d+) negative .* min=(?P<min>-?\d+), max=(?P<max>-?\d+)$"
)
RE_FINAL_RESULT_START = re.compile(r"^#+Final Result#+$")
RE_FINAL_RESULT_END = re.compile(r"^#+$")
RE_KEYED_DICT = re.compile(r"^(?P<key>[A-Za-z0-9_]+)\s*:\s*(?P<value>\{.*\})\s*$")

KNOWN_METHODS = {
    "omd_tch_eg",
    "omd_tch_pgd",
    "eu",
    "eu_fast",
    "omd_tch",
    "afleg",
    "afl",
    "ada_omd_tch_eg",
    "ada_afleg",
}


def infer_method_from_path(log_path: Path) -> Optional[str]:
    for part in log_path.parts:
        if part in KNOWN_METHODS:
            return part
    return None


def infer_seed_train_from_path(log_path: Path) -> Tuple[Optional[int], Optional[int]]:
    for part in log_path.parts:
        match = re.match(r"^seed_(\d+)_train_(\d+)$", part)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def safe_float(value: str) -> float:
    return float(value.replace(",", ""))


def parse_jsonish_dict(value: str) -> Any:
    return ast.literal_eval(value)


def finalize_epoch(epoch_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Drop internal scratch keys and normalize missing values."""
    cleaned = dict(epoch_entry)
    train_loss = cleaned.pop("_train_loss_avg", None)
    if cleaned.get("train_loss") is None and train_loss is not None:
        cleaned["train_loss"] = train_loss

    # Keep a stable schema with nullable fields.
    for key in [
        "train_acc",
        "train_loss",
        "grad_norm_avg",
        "retain_acc",
        "forget_acc",
        "test_acc",
        "forget_loss",
        "retain_loss",
        "param_update_norm",
        "duration_sec",
    ]:
        cleaned.setdefault(key, None)
    return cleaned


def parse_run_log(log_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    method = infer_method_from_path(log_path)
    seed, train_seed = infer_seed_train_from_path(log_path)

    training_log: Dict[str, Any] = {
        "parsed_from_log": True,
        "method": method,
        "seed": seed,
        "train_seed": train_seed,
        "epochs": [],
        "final_validation": {},
    }
    evaluation_result: Dict[str, Any] = {"parsed_from_log": True}

    current_epoch: Optional[Dict[str, Any]] = None
    final_validation_raw: Dict[str, float] = {}
    final_validation_loss: Dict[str, float] = {}
    forget_set_info: Dict[str, Any] = {}
    forget_target_integrity: Dict[str, Any] = {}
    in_final_result = False
    in_final_eval_block = False

    def flush_current_epoch() -> None:
        nonlocal current_epoch
        if current_epoch is not None:
            training_log["epochs"].append(finalize_epoch(current_epoch))
            current_epoch = None

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        if RE_FINAL_RESULT_START.match(line):
            in_final_result = True
            continue
        if in_final_result and line.startswith("#") and not line.startswith("############################Final Result"):
            # Some logs use a separator line after the block. Once we've entered
            # the final result block, stop collecting on a long hash divider.
            break

        match = RE_SEED.match(line)
        if match:
            if seed is None:
                seed = int(match.group("seed"))
                training_log["seed"] = seed
            continue

        match = RE_FORGET_SET_INFO.match(line)
        if match:
            forget_set_info = {
                "size": int(match.group("size")),
                "retain_size": None,
                "class_distribution": parse_jsonish_dict(match.group("dist")),
            }
            training_log["forget_set_info"] = forget_set_info
            continue

        match = RE_RETAIN_FORGET_COUNTS.match(line)
        if match:
            kind = match.group("kind")
            count = int(match.group("count"))
            forget_set_info = training_log.get("forget_set_info", {})
            if kind == "retain":
                forget_set_info["retain_size"] = count
            else:
                forget_set_info["size"] = count
            training_log["forget_set_info"] = forget_set_info
            continue

        match = RE_EPOCH_START.match(line)
        if match:
            flush_current_epoch()
            current_epoch = {
                "epoch": int(match.group("epoch")),
                "lr": safe_float(match.group("lr")),
                "duration_sec": None,
                "train_acc": None,
                "train_loss": None,
                "grad_norm_avg": None,
                "retain_acc": None,
                "forget_acc": None,
                "test_acc": None,
                "forget_loss": None,
                "retain_loss": None,
                "param_update_norm": None,
            }
            continue

        if current_epoch is not None:
            match = RE_TRAIN_BATCH.match(line)
            if match and int(match.group("epoch")) == current_epoch["epoch"]:
                current_epoch["_train_loss_avg"] = safe_float(match.group("loss_avg"))
                continue

            match = RE_TRAIN_ACC.match(line)
            if match:
                current_epoch["train_acc"] = safe_float(match.group("train_acc"))
                continue

            match = RE_DURATION.match(line)
            if match:
                current_epoch["duration_sec"] = safe_float(match.group("duration_sec"))
                continue

            match = RE_EPOCH_SUMMARY.match(line)
            if match and int(match.group("epoch")) == current_epoch["epoch"]:
                current_epoch["retain_acc"] = safe_float(match.group("retain_acc"))
                current_epoch["forget_acc"] = safe_float(match.group("forget_acc"))
                current_epoch["test_acc"] = safe_float(match.group("test_acc"))
                current_epoch["forget_loss"] = safe_float(match.group("forget_loss"))
                current_epoch["retain_loss"] = safe_float(match.group("retain_loss"))
                current_epoch["param_update_norm"] = safe_float(match.group("param_update_norm"))
                continue

            match = RE_RL_REWARD.match(line)
            if match:
                current_epoch["rl_reward"] = {
                    "mean": safe_float(match.group("mean")),
                    "std": safe_float(match.group("std")),
                    "min": safe_float(match.group("min")),
                    "max": safe_float(match.group("max")),
                }
                continue

            match = RE_RL_OVERLAP.match(line)
            if match:
                current_epoch["rl_label_overlap"] = safe_float(match.group("overlap"))
                current_epoch["rl_label_overlap_ratio"] = safe_float(match.group("ratio"))
                continue

        match = RE_FORGET_TARGET_INTEGRITY.match(line)
        if match:
            forget_target_integrity = {
                "negative_count_at_eval": int(match.group("negative_count")),
                "target_min": int(match.group("min")),
                "target_max": int(match.group("max")),
            }
            training_log["forget_target_integrity"] = forget_target_integrity
            in_final_eval_block = True
            continue

        if in_final_eval_block:
            match = RE_VALID_BATCH.match(line)
            if match:
                split = match.group("split")
                final_validation_loss[split] = safe_float(match.group("loss_avg"))
                continue

            match = RE_VALID_RAW_ACC.match(line)
            if match:
                split = match.group("split")
                final_validation_raw[split] = safe_float(match.group("acc"))
                continue

            match = RE_VALID_FINAL_RAW_ACC.match(line)
            if match:
                split = match.group("split")
                final_validation_raw.setdefault(split, safe_float(match.group("acc")))
                continue

        if in_final_result:
            match = RE_KEYED_DICT.match(line)
            if match:
                key = match.group("key")
                value = parse_jsonish_dict(match.group("value"))
                evaluation_result[key] = value
            continue

    flush_current_epoch()

    # Final validation is reconstructed from the explicit raw accuracy lines.
    if final_validation_raw:
        final_validation: Dict[str, Dict[str, Any]] = {}
        for split, raw_acc in final_validation_raw.items():
            reported = round(100.0 - raw_acc, 2) if split == "forget" else round(raw_acc, 2)
            final_validation[split] = {
                "accuracy": raw_acc,
                "loss": final_validation_loss.get(split),
                "reported_accuracy": reported,
            }
        training_log["final_validation"] = final_validation

    # Fallback: if the run.log did not contain a keyed final result block, derive the
    # primary accuracy dictionary from final validation.
    if "accuracy" not in evaluation_result and final_validation_raw:
        evaluation_result["accuracy"] = {
            split: (round(100.0 - raw_acc, 2) if split == "forget" else round(raw_acc, 2))
            for split, raw_acc in final_validation_raw.items()
        }

    # Keep evaluation_result.json structurally useful even when some optional blocks
    # were not present in the log.
    if forget_target_integrity and "forget_target_integrity" not in training_log:
        training_log["forget_target_integrity"] = forget_target_integrity

    return evaluation_result, training_log


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_log", type=Path, help="Path to run.log to parse")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for JSON files. Defaults to the run.log directory.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_log = args.run_log.expanduser().resolve()
    if not run_log.exists():
        raise SystemExit(f"Run log not found: {run_log}")

    out_dir = args.out_dir.expanduser().resolve() if args.out_dir is not None else run_log.parent
    evaluation_result, training_log = parse_run_log(run_log)

    write_json(out_dir / "evaluation_result.json", evaluation_result)
    write_json(out_dir / "training_log.json", training_log)
    write_json(out_dir / "train_log.json", training_log)

    print(f"Wrote: {out_dir / 'evaluation_result.json'}")
    print(f"Wrote: {out_dir / 'training_log.json'}")
    print(f"Wrote: {out_dir / 'train_log.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
