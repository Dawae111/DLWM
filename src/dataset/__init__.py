"""Dataset factories with lazy imports.

Keeping this module lazy prevents an optional GRPO or packed-data dependency
from breaking unrelated SFT/DLWM entry points at import time.
"""

from importlib import import_module


_FACTORIES = {
    "make_dpo_data_module": ("src.dataset.dpo_dataset", "make_dpo_data_module"),
    "make_grpo_data_module": ("src.dataset.grpo_dataset", "make_grpo_data_module"),
    "make_supervised_data_module": (
        "src.dataset.sft_dataset",
        "make_supervised_data_module",
    ),
    "make_supervised_data_module_lvr": (
        "src.dataset.lvr_sft_dataset",
        "make_supervised_data_module_lvr",
    ),
    "make_packed_supervised_data_module_lvr": (
        "src.dataset.lvr_sft_dataset_packed",
        "make_packed_supervised_data_module_lvr",
    ),
}

__all__ = list(_FACTORIES)


def __getattr__(name):
    if name not in _FACTORIES:
        raise AttributeError(name)
    module_name, attribute = _FACTORIES[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
