"""Compatibility shim for `scripts.train`."""

from scripts.train import *  # noqa: F401,F403


if __name__ == "__main__":
    from scripts.train import main

    main()
