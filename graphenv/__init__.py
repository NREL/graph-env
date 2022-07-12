from types import ModuleType
from typing import Optional

from . import _version

__version__ = _version.get_versions()["version"]

from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf: Optional[ModuleType] = None
tf1: Optional[ModuleType] = None

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

if tf1 is not None:
    assert tfv == 2
    assert tf is not None
    if not tf1.executing_eagerly():
        tf.compat.v1.enable_eager_execution()
