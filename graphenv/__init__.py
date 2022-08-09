from ray.rllib.utils.framework import try_import_tf, try_import_torch

try:
    from graphenv._version import __version__, __version_tuple__  # noqa: F401
except ModuleNotFoundError:
    __version__ = "package not installed"
    __version_tuple__ = (None, None, None)

tf1, tf, tfv = try_import_tf()
assert tfv == 2
if not tf1.executing_eagerly():
    tf1.enable_eager_execution()

torch, nn = try_import_torch()
