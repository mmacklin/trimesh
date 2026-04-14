from . import ray_triangle

# optionally load an interface to the embree raytracer
try:
    from . import ray_pyembree

    has_embree = True
except BaseException as E:
    from .. import exceptions

    ray_pyembree = exceptions.ExceptionWrapper(E)
    has_embree = False

# check if Warp is available without importing it eagerly
# (importing ray_warp triggers Warp initialization + kernel JIT)
try:
    from importlib.util import find_spec as _find_spec

    has_warp = _find_spec("warp") is not None
except BaseException:
    has_warp = False

# ray_warp is loaded lazily on first access to avoid
# Warp initialization cost at `import trimesh` time
_ray_warp_module = None


def _load_ray_warp():
    """Lazily import ray_warp and cache the module."""
    global _ray_warp_module, has_warp, ray_warp
    if _ray_warp_module is not None:
        return _ray_warp_module
    try:
        import importlib

        _ray_warp_module = importlib.import_module(".ray_warp", __package__)
        ray_warp = _ray_warp_module
        return _ray_warp_module
    except BaseException:
        has_warp = False
        return None


# placeholder so attribute access doesn't raise before lazy load
ray_warp = None

# add to __all__ as per pep8
__all__ = ["ray_pyembree", "ray_triangle", "ray_warp"]
