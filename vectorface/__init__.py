from .base import Vector, BoolVector, IntVector, FloatVector, VectorBackend
from .python_backend import PYTHON_BACKEND

__all__ = [
    '__version__',
    'Vector',
    'BoolVector',
    'IntVector',
    'FloatVector',
    'VectorBackend',
    'list_available_backends',
    'backend_is_available',
    'get_backend',
    'activate_backend',
    'register_backend',
]

__version__ = '0.0'


BACKENDS = dict(python=PYTHON_BACKEND)

try:
    from .numpy_backend import NUMPY_BACKEND
    BACKENDS.update(numpy=NUMPY_BACKEND)
except ImportError:
    NUMPY_BACKEND = None


PYTHON_BACKEND.activate()


def list_available_backends() -> list[str]:
    return sorted(BACKENDS)


def backend_is_available(name: str) -> bool:
    return name in BACKENDS


def get_backend(name: str) -> VectorBackend:
    if name not in BACKENDS:
        raise RuntimeError(f"The {name} backend is not available.")
    return BACKENDS[name]


def activate_backend(name: str) -> None:
    get_backend(name).activate()


def register_backend(name: str, obj: VectorBackend) -> None:
    if name in BACKENDS:
        raise RuntimeError(f"The {name} backend is already registered.")
    BACKENDS[name] = obj


# TODO: Contextual backend selection:
#       with default_backend('python') as backend:
#           ...
