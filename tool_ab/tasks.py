"""Write the fixed task order for the tool A/B runs to configs/tasks.txt.

    python tool_ab/tasks.py

All 500 SWE-bench Verified tasks, shuffled once with a fixed seed. Every arm runs a prefix
of this list (`run_arm.py --slice 0:N`), so arms always share their tasks and a larger
budget extends the experiment without discarding earlier runs.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

DATASET = "princeton-nlp/SWE-Bench_Verified"


def ordered_tasks(task_ids: list[str], seed: int) -> list[str]:
    tasks = sorted(task_ids)
    random.Random(seed).shuffle(tasks)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "configs" / "tasks.txt")
    args = parser.parse_args()

    from datasets import load_dataset

    ids = [row["instance_id"] for row in load_dataset(DATASET, split="test")]
    tasks = ordered_tasks(ids, args.seed)
    header = f"# {DATASET} test split, {len(tasks)} tasks, shuffled with seed {args.seed}\n"
    args.out.write_text(header + "\n".join(tasks) + "\n")
    print(f"wrote {len(tasks)} tasks to {args.out}")


if __name__ == "__main__":
    main()
