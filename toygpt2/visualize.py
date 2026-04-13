"""Compatibility shim for `scripts.visualize`."""

from scripts.visualize import *  # noqa: F401,F403


if __name__ == "__main__":
    from scripts.visualize import main

    main()
