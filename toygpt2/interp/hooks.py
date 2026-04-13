"""PyTorch hook helpers for quick interpretability experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch


@dataclass
class HookCollection:
    """Keep track of registered hooks so they can be removed cleanly."""

    handles: List[torch.utils.hooks.RemovableHandle]
    cache: Dict[str, object]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def register_output_hooks(
    model: torch.nn.Module,
    module_name_filters: Sequence[str] | None = None,
) -> HookCollection:
    """Capture module inputs and outputs by module name."""

    cache: Dict[str, object] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def should_capture(name: str) -> bool:
        if module_name_filters is None:
            return True
        return any(fragment in name for fragment in module_name_filters)

    for name, module in model.named_modules():
        if not name or not should_capture(name):
            continue

        def hook_fn(module: torch.nn.Module, inputs: tuple[object, ...], output: object, name: str = name) -> None:
            captured_inputs = []
            for value in inputs:
                if torch.is_tensor(value):
                    captured_inputs.append(value.detach())
                else:
                    captured_inputs.append(value)
            if torch.is_tensor(output):
                captured_output = output.detach()
            else:
                captured_output = output
            cache[name] = {"inputs": captured_inputs, "output": captured_output}

        handles.append(module.register_forward_hook(hook_fn))

    return HookCollection(handles=handles, cache=cache)
