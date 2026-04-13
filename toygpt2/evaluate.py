"""Compatibility shim for `scripts.evaluate`."""

from scripts.evaluate import *  # noqa: F401,F403


if __name__ == "__main__":
    from scripts.evaluate import main

    main()
