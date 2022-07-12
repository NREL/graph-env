from types import ModuleType
from typing import Optional

from . import _version

__version__ = _version.get_versions()["version"]

from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf: Optional[ModuleType] = None
tf1: Optional[ModuleType] = None

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
