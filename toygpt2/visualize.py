"""Plot loss curves from JSONL training logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(log_path: str | Path) -> List[Dict[str, float]]:
    with Path(log_path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _series(records: Sequence[Dict[str, float]], split: str) -> tuple[List[int], List[float]]:
    filtered = [record for record in records if record.get("split") == split]
    return [int(record["step"]) for record in filtered], [float(record["loss"]) for record in filtered]


def plot_logs(
    log_paths: Sequence[str | Path],
    output_path: str | Path,
    labels: Sequence[str] | None = None,
    compare_split: str | None = None,
    title: str | None = None,
) -> str:
    """Plot one run or overlay several runs."""

    if labels is None:
        labels = [Path(path).parent.name for path in log_paths]

    figure, axis = plt.subplots(figsize=(8, 5))

    if len(log_paths) == 1 and compare_split is None:
        records = load_metrics(log_paths[0])
        for split, style in [("train", "-"), ("val", "--")]:
            steps, losses = _series(records, split)
            if steps:
                axis.plot(steps, losses, linestyle=style, label=f"{labels[0]} {split}")
    else:
        if compare_split is None:
            compare_split = "val"
        for label, log_path in zip(labels, log_paths):
            steps, losses = _series(load_metrics(log_path), compare_split)
            if steps:
                axis.plot(steps, losses, label=f"{label} {compare_split}")

    axis.set_xlabel("Step")
    axis.set_ylabel("Loss")
    axis.set_title(title or "Loss Curves")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return str(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training/validation loss curves from JSONL logs.")
    parser.add_argument("--logs", nargs="+", required=True, help="One or more metrics.jsonl files.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels for the logs.")
    parser.add_argument("--output", type=str, required=True, help="Output image path.")
    parser.add_argument("--compare_split", type=str, default=None, choices=["train", "val"], help="Overlay a single split when plotting multiple logs.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_path = plot_logs(
        log_paths=args.logs,
        labels=args.labels,
        output_path=args.output,
        compare_split=args.compare_split,
        title=args.title,
    )
    print(plot_path)


if __name__ == "__main__":
    main()
