# mini_gpt_attnres

Clean, minimal decoder-only language-model project with two variants:

- `StandardGPT`: standard GPT-2 style baseline
- `AttnResGPT`: GPT-style decoder with Kimi-style Attention Residuals over residual-history sites

## Features

- Small default model sizes for CPU/GPU smoke tests
- Shared training/evaluation/checkpoint interfaces for both variants
- Structured `ResidualHistory` with explicit `site_id` and `layer_id`
- JSONL training logs plus matplotlib plotting
- Forward caches for attention internals and AttnRes site internals
- Simple hook, ablation, and patching helpers

## Install

```bash
pip install -r mini_gpt_attnres/requirements.txt
```

## Quick Start

Forward demo:

```bash
python -m mini_gpt_attnres.demo_forward
```

Train the baseline:

```bash
python -m mini_gpt_attnres.train --model_type standard --max_steps 50 --out_dir mini_gpt_attnres_runs/standard
```

Train the AttnRes variant:

```bash
python -m mini_gpt_attnres.train --model_type attnres --max_steps 50 --out_dir mini_gpt_attnres_runs/attnres
```

Evaluate a checkpoint:

```bash
python -m mini_gpt_attnres.evaluate --checkpoint mini_gpt_attnres_runs/attnres/ckpt_last.pt
```

Plot a loss curve:

```bash
python -m mini_gpt_attnres.visualize --logs mini_gpt_attnres_runs/attnres/metrics.jsonl --output mini_gpt_attnres_runs/attnres/loss.png
```

Run a tiny baseline-vs-AttnRes comparison:

```bash
python -m mini_gpt_attnres.demo_compare
```

Run a TinyStories smoke test (small subset, short training):

```bash
python -m mini_gpt_attnres.run_tinystories_smoke \
  --train_texts 2000 \
  --val_texts 200 \
  --max_steps 20 \
  --out_dir mini_gpt_attnres_runs/tinystories_smoke
```

## AttnRes Design

Each `AttnResGPT` block has two residual-history sites:

- `blocks.{i}.pre_attn`
- `blocks.{i}.pre_mlp`

For each site:

1. Stack `history + [current]` along the site dimension.
2. Normalize over the hidden dimension.
3. Score each site using a learned static query.
4. Softmax over the site dimension.
5. Weighted-sum the residual states.

The aggregated `pre_attn` state feeds `ln_1 -> self_attn`, and the aggregated `pre_mlp` state feeds `ln_2 -> mlp`.

## Cache Keys

Both models expose:

- `embedding_out`
- `blocks.{i}.input`
- `blocks.{i}.attn_out`
- `blocks.{i}.mlp_out`
- `blocks.{i}.output`
- `blocks.{i}.attn_probs`
- `blocks.{i}.q`
- `blocks.{i}.k`
- `blocks.{i}.v`

`AttnResGPT` additionally exposes:

- `blocks.{i}.attnres.pre_attn.current`
- `blocks.{i}.attnres.pre_attn.history_states`
- `blocks.{i}.attnres.pre_attn.scores`
- `blocks.{i}.attnres.pre_attn.weights`
- `blocks.{i}.attnres.pre_attn.aggregated`
- `blocks.{i}.attnres.pre_mlp.current`
- `blocks.{i}.attnres.pre_mlp.history_states`
- `blocks.{i}.attnres.pre_mlp.scores`
- `blocks.{i}.attnres.pre_mlp.weights`
- `blocks.{i}.attnres.pre_mlp.aggregated`
