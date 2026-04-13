"""Compatibility shim for `scripts.demo_compare`."""

from scripts.demo_compare import *  # noqa: F401,F403


if __name__ == "__main__":
    from scripts.demo_compare import main

    main()
