from . import _version

__version__ = _version.get_versions()["version"]

from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
assert tfv == 2
if not tf1.executing_eagerly():
    tf1.enable_eager_execution()

torch, nn = try_import_torch()
