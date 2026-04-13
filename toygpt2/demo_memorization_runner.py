"""Compatibility shim for `scripts.demo_memorization_runner`."""

from scripts.demo_memorization_runner import *  # noqa: F401,F403


if __name__ == "__main__":
    from scripts.demo_memorization_runner import main

    main()
