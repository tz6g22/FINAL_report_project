"""Launch two concurrent 2-GPU DDP jobs (standard + attnres)."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
from copy import deepcopy
from pathlib import Path
from typing import List, Sequence


def _append_optional_arg(cmd: List[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _build_train_command(
    model_type: str,
    out_dir: Path,
    master_port: int,
    nproc_per_node: int,
    args: argparse.Namespace,
    extra_train_args: Sequence[str],
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc_per_node),
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        str(master_port),
        "-m",
        "scripts.train",
        "--model_type",
        model_type,
        "--out_dir",
        str(out_dir),
    ]
    _append_optional_arg(cmd, "--dataset_type", args.dataset_type)
    _append_optional_arg(cmd, "--dataset_name", args.dataset_name)
    _append_optional_arg(cmd, "--tokenizer_name", args.tokenizer_name)
    _append_optional_arg(cmd, "--train_texts", args.train_texts)
    _append_optional_arg(cmd, "--val_texts", args.val_texts)
    _append_optional_arg(cmd, "--max_steps", args.max_steps)
    _append_optional_arg(cmd, "--batch_size", args.batch_size)
    _append_optional_arg(cmd, "--block_size", args.block_size)
    _append_optional_arg(cmd, "--block_stride", args.block_stride)
    _append_optional_arg(cmd, "--eval_interval", args.eval_interval)
    _append_optional_arg(cmd, "--checkpoint_interval", args.checkpoint_interval)
    _append_optional_arg(cmd, "--eval_batches", args.eval_batches)
    _append_optional_arg(cmd, "--learning_rate", args.learning_rate)
    _append_optional_arg(cmd, "--n_layer", args.n_layer)
    _append_optional_arg(cmd, "--n_head", args.n_head)
    _append_optional_arg(cmd, "--n_embd", args.n_embd)
    _append_optional_arg(cmd, "--num_workers", args.num_workers)
    _append_optional_arg(cmd, "--seed", args.seed)
    _append_optional_arg(cmd, "--device", args.device)
    cmd.extend(extra_train_args)
    return cmd


def _stream_prefixed_output(prefix: str, stream) -> None:
    for line in iter(stream.readline, ""):
        if not line:
            break
        rendered = line.rstrip()
        expected = f"[{prefix}] "
        if rendered.startswith(expected):
            print(rendered, flush=True)
        else:
            print(f"[{prefix}] {rendered}", flush=True)
    stream.close()


def _prepare_tinystories_token_cache_once(args: argparse.Namespace) -> None:
    if args.dataset_type != "tinystories":
        return

    from toygpt2.config import default_experiment
    from data.data_tinystories import prepare_tinystories_assets

    experiment = default_experiment(model_type="standard")
    experiment.data.dataset_type = args.dataset_type
    if args.dataset_name is not None:
        experiment.data.hf_dataset_name = args.dataset_name
    if args.tokenizer_name is not None:
        experiment.data.tokenizer_name = args.tokenizer_name
    if args.train_texts is not None:
        experiment.data.train_texts = args.train_texts
    if args.val_texts is not None:
        experiment.data.val_texts = args.val_texts
    if args.block_stride is not None:
        experiment.data.block_stride = args.block_stride
    if args.block_size is not None:
        experiment.model.block_size = args.block_size

    print("[launcher] preparing TinyStories token cache (single tokenization pass)...", flush=True)
    standard_model_config = deepcopy(experiment.model)
    standard_model_config.model_type = "standard"
    prepare_tinystories_assets(
        model_config=standard_model_config,
        data_config=experiment.data,
        verbose=True,
        allow_cache_build=True,
    )

    attnres_model_config = deepcopy(standard_model_config)
    attnres_model_config.model_type = "attnres"
    prepare_tinystories_assets(
        model_config=attnres_model_config,
        data_config=experiment.data,
        verbose=True,
        allow_cache_build=False,
    )
    print("[launcher] token cache ready for both standard and attnres.", flush=True)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch concurrent 2x2-GPU DDP training (standard + attnres).")
    parser.add_argument("--dataset_type", type=str, default="tinystories")
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--train_texts", type=int, default=None, help="Optional train-text cap. Default uses full train split.")
    parser.add_argument("--val_texts", type=int, default=None, help="Optional val-text cap. Default uses full validation split.")
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=32, help="Per-GPU batch size in each DDP job.")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--block_stride", type=int, default=256)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=None, help="Optional eval batch cap. Default evaluates full val split.")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda", help="Passed to train.py.")
    parser.add_argument("--base_out_dir", type=str, default="toygpt2_runs/tinystories_dual")
    parser.add_argument("--nproc_per_model", type=int, default=2)
    parser.add_argument("--standard_gpus", type=str, default="0,1")
    parser.add_argument("--attnres_gpus", type=str, default="2,3")
    parser.add_argument("--standard_master_port", type=int, default=29510)
    parser.add_argument("--attnres_master_port", type=int, default=29520)
    parser.add_argument("--dry_run", action="store_true", help="Print subprocess commands without launching.")
    return parser.parse_known_args()


def main() -> None:
    args, extra_train_args = parse_args()
    base_out_dir = Path(args.base_out_dir).expanduser().resolve()
    standard_out_dir = base_out_dir / "standard"
    attnres_out_dir = base_out_dir / "attnres"
    standard_out_dir.mkdir(parents=True, exist_ok=True)
    attnres_out_dir.mkdir(parents=True, exist_ok=True)

    standard_cmd = _build_train_command(
        model_type="standard",
        out_dir=standard_out_dir,
        master_port=args.standard_master_port,
        nproc_per_node=args.nproc_per_model,
        args=args,
        extra_train_args=extra_train_args,
    )
    attnres_cmd = _build_train_command(
        model_type="attnres",
        out_dir=attnres_out_dir,
        master_port=args.attnres_master_port,
        nproc_per_node=args.nproc_per_model,
        args=args,
        extra_train_args=extra_train_args,
    )

    print("[launcher] standard cmd:", shlex.join(standard_cmd), flush=True)
    print("[launcher] attnres cmd:", shlex.join(attnres_cmd), flush=True)
    if args.dry_run:
        return

    _prepare_tinystories_token_cache_once(args)

    env_standard = os.environ.copy()
    env_attnres = os.environ.copy()
    env_standard["CUDA_VISIBLE_DEVICES"] = args.standard_gpus
    env_attnres["CUDA_VISIBLE_DEVICES"] = args.attnres_gpus
    env_standard.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    env_attnres.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    env_standard.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env_attnres.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    processes = {}
    threads: list[threading.Thread] = []
    try:
        processes["standard"] = subprocess.Popen(
            standard_cmd,
            env=env_standard,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes["attnres"] = subprocess.Popen(
            attnres_cmd,
            env=env_attnres,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for name, process in processes.items():
            if process.stdout is None:
                continue
            thread = threading.Thread(
                target=_stream_prefixed_output,
                args=(name, process.stdout),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        exit_codes = {name: proc.wait() for name, proc in processes.items()}
        for thread in threads:
            thread.join()

        failed = {name: code for name, code in exit_codes.items() if code != 0}
        if failed:
            raise SystemExit(f"[launcher] one or more jobs failed: {failed}")
        print("[launcher] both DDP jobs completed successfully.", flush=True)
    except KeyboardInterrupt:
        print("[launcher] interrupted, terminating child processes...", flush=True)
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        raise


if __name__ == "__main__":
    main()
