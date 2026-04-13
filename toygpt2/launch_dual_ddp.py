"""Compatibility shim for `scripts.launch_dual_ddp`."""

from scripts.launch_dual_ddp import *  # noqa: F401,F403


if __name__ == "__main__":
    from scripts.launch_dual_ddp import main

    main()
