"""Compatibility launcher for the veRL OPD training recipe.

Prefer ``scripts/run_opd.sh``. This wrapper keeps a Python entry point for job
schedulers while delegating all training to ``verl.trainer.main_ppo``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", "--train_file", required=True)
    parser.add_argument("--output-dir", "--output_dir", default="outputs/opd")
    parser.add_argument("--model", "--model_name_or_path")
    parser.add_argument("--teacher-model", "--teacher_model_name_or_path")
    parser.add_argument("--lora-init-path", "--lora_init_path")
    args, overrides = parser.parse_known_args()

    env = os.environ.copy()
    if args.model:
        env["MODEL_PATH"] = args.model
    if args.teacher_model:
        env["TEACHER_MODEL_PATH"] = args.teacher_model
    if args.lora_init_path:
        env["LORA_INIT_PATH"] = args.lora_init_path
    script = Path(__file__).resolve().parent / "scripts" / "run_opd.sh"
    os.execve(script, [str(script), args.train_file, args.output_dir, *overrides], env)


if __name__ == "__main__":
    main()
