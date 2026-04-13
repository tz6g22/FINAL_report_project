"""Compatibility shim for `scripts.run_tinystories`."""

from scripts.run_tinystories import *  # noqa: F401,F403


if __name__ == "__main__":
    from scripts.run_tinystories import main

    main()
