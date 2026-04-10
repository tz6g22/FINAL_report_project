"""Compatibility stub: TinyStories smoke mode is disabled."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "run_tinystories_smoke is disabled. "
        "Use `python -m mini_gpt_attnres.run_tinystories` for formal TinyStories training."
    )


if __name__ == "__main__":
    main()
