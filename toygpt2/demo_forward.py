"""Compatibility shim for `scripts.demo_forward`."""

from scripts.demo_forward import *  # noqa: F401,F403


if __name__ == "__main__":
    from scripts.demo_forward import main

    main()
