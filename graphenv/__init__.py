from ray.rllib.utils.framework import try_import_tf, try_import_torch

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown version"
    __version_tuple__ = (0, 0, "unknown version")

tf1, tf, tfv = try_import_tf()

if tf is not None:
    assert tfv == 2, "GraphEnv only supports tensorflow 2.x"
if not tf1.executing_eagerly():
    tf1.enable_eager_execution()

torch, nn = try_import_torch()
