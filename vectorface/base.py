import inspect
import logging
from abc import abstractmethod, ABC, ABCMeta
from dataclasses import dataclass, field
from functools import reduce
from typing import TypeVar, Optional, Sequence, Iterable, Callable, Iterator, Any, Generic, overload, Hashable, Mapping

__all__ = [
    'BackendVector',
    'BoolBackendVector',
    'IntBackendVector',
    'FloatBackendVector',
    'Vector',
    'BoolVector',
    'IntVector',
    'FloatVector',
    'VectorBackend',
]

_BackendValue = TypeVar('_BackendValue', bound=Hashable)
_BackendValue2 = TypeVar('_BackendValue2', bound=Hashable)
_BackendValue3 = TypeVar('_BackendValue3', bound=Hashable)
_BackendBool = TypeVar('_BackendBool', bound=Hashable)
_FrontendValue = TypeVar('_FrontendValue', bound=Hashable)
_FrontendValue2 = TypeVar('_FrontendValue2', bound=Hashable)
_FrontendValue3 = TypeVar('_FrontendValue3', bound=Hashable)

_ValueType = TypeVar('_ValueType', bound=type[Hashable])
_ValueType2 = TypeVar('_ValueType2', bound=type[Hashable])

_Method = TypeVar('_Method', bound=Callable)

LOGGER = logging.getLogger(__name__)


# @dataclass
# class DistributionFamilyParam(Generic[_ValueType]):
#     name: str
#     value_type: type[_ValueType] | str
#     default: Optional[_ValueType]
#
#     def __post_init__(self):
#         if isinstance(self.value_type, type):
#             assert self.default is None or isinstance(self.default, self.value_type)
#         else:
#             assert self.value_type == 'reflexive'
#             assert self.default is None
#
#
# @dataclass
# class DistributionFamily(Generic[_ValueType]):
#     name: str
#     value_types: set[type[_ValueType]] = field(default_factory=set)
#     params: dict[str, DistributionFamilyParam] = field(default_factory=dict)
#
#
# DIST_FAMILY_REGISTRY: dict[str, DistributionFamily] = {}
#
#
# def register_dist_family(name: str):
#     assert isinstance(name, str)
#     if name not in DIST_FAMILY_REGISTRY:
#         DIST_FAMILY_REGISTRY[name] = DistributionFamily(name)
#
#
# def register_dist_family_value_type(name: str, value_type: type[_FrontendValue]):
#     assert isinstance(value_type, type)
#     register_dist_family(name)
#     DIST_FAMILY_REGISTRY[name].value_types.add(value_type)
#
#
# def register_dist_family_param(name: str, param_name: str, param_type: type[_ValueType] | str = 'reflexive',
#                                default_value: _ValueType = None):
#     register_dist_family(name)
#     dist_family = DIST_FAMILY_REGISTRY[name]
#     param = DistributionFamilyParam(param_name, param_type, default_value)
#     if param_name in dist_family.params:
#         assert dist_family.params[param_name] == param
#     else:
#         dist_family.params[param_name] = param


class BackendVectorMeta(ABCMeta):

    def __new__(cls, name, bases, dct):
        vector_class = super().__new__(cls, name, bases, dct)
        assert name == 'BackendVector' or issubclass(vector_class, BackendVector)
        if not inspect.isabstract(vector_class):
            vector_class.get_backend().register(vector_class)
        return vector_class


def unimplemented(*_args, **_kwargs):
    print(_args, _kwargs)
    raise NotImplementedError()


@dataclass
class OptionalMethod(Generic[_Method]):
    prototype: _Method
    implementation: _Method = unimplemented
    instance: Any = None

    def __get__(self, instance, owner):
        return OptionalMethod(self.prototype, self.implementation, instance)

    def __call__(self, *args, **kwargs):
        return self.implementation(self.instance, *args, **kwargs)

    @property
    def is_defined(self) -> bool:
        return self.implementation is not unimplemented

    def define(self, implementation: _Method) -> 'OptionalMethod[_Method]':
        if implementation.__name__ != self.prototype.__name__:
            raise AttributeError("Implementation name must match original method name.")
        return OptionalMethod(self.prototype, implementation)

    def __getattr__(self, item):
        if self.implementation is not unimplemented:
            try:
                return getattr(self.implementation, item)
            except AttributeError:
                pass
        return getattr(self.prototype, item)

    def __setattr__(self, key, value):
        if key in ('prototype', 'implementation', 'instance'):
            super().__setattr__(key, value)
            return
        if self.implementation is not unimplemented:
            setattr(self.implementation, key, value)
            return
        raise AttributeError(f"Attribute {key} is not settable.")


def optional_method(method: _Method) -> OptionalMethod[_Method]:
    return OptionalMethod(method)


class BackendVector(Sequence[_BackendValue], ABC, metaclass=BackendVectorMeta):
    _hash_value: Optional[int] = None

    @classmethod
    @abstractmethod
    def new(cls, value_type: type[_BackendValue], values: Iterable[_ValueType]) -> 'BackendVector[_BackendValue]':
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def sample_uniform(cls, value_type: type[_BackendValue], length: int, *, minimum: _BackendValue,
                       maximum: _BackendValue) -> 'BackendVector[_BackendValue]':
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def sample_categorical(cls, value_type: type[_BackendValue], length: int, *,
                           weights: dict[_BackendValue, float]) -> 'BackendVector[_BackendValue]':
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def sample_gaussian(cls, value_type: type[_BackendValue], length: int, *, mean: _BackendValue,
                        stddev: _BackendValue) -> 'BackendVector[_BackendValue]':
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_backend(cls) -> 'VectorBackend':
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_supported_value_types(cls) -> tuple[type[_BackendValue2], ...]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def is_sparse(cls) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def value_type(self) -> type[_BackendValue]:
        raise NotImplementedError()

    @abstractmethod
    def cast(self, value_type: type[_BackendValue2]) -> 'BackendVector[_BackendValue2]':
        raise NotImplementedError()

    @abstractmethod
    def all(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def any(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def count(self, value: _BackendValue) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @overload
    def __getitem__(self, index: int) -> '_BackendValue': ...

    @overload
    def __getitem__(self, index: slice) -> 'BackendVector[_BackendValue]': ...

    @abstractmethod
    def __getitem__(self, index: int | slice) -> '_BackendValue | Vector[_BackendValue]':
        raise NotImplementedError()

    @abstractmethod
    def reversed(self) -> 'BackendVector[_BackendValue]':
        raise NotImplementedError()

    @abstractmethod
    def concat(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[_BackendValue]':
        raise NotImplementedError()

    @abstractmethod
    def eq(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[_BackendBool]':
        raise NotImplementedError()

    @abstractmethod
    def ne(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[_BackendBool]':
        raise NotImplementedError()

    @optional_method
    def le(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[_BackendBool]': ...

    @optional_method
    def ge(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[_BackendBool]': ...

    @optional_method
    def lt(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[_BackendBool]': ...

    @optional_method
    def gt(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[_BackendBool]': ...

    @optional_method
    def abs(self) -> 'BackendVector[_BackendValue]': ...

    @optional_method
    def add(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[_BackendValue]': ...

    @optional_method
    def sub(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[_BackendValue]': ...

    @optional_method
    def mul(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[_BackendValue]': ...

    @optional_method
    def div(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[_BackendValue]': ...

    def nonzero(self) -> 'BackendVector[_BackendBool]':
        return self.cast(self.get_backend().get_default_backend_type(bool))

    def __bool__(self) -> bool:
        return self.any()

    def __str__(self) -> str:
        return repr(tuple(self))

    def __repr__(self) -> str:
        return type(self).__name__ + '.new(' + self.value_type.__name__ + ', ' + repr(tuple(self)) + ')'

    def __iter__(self) -> Iterator[_BackendValue]:
        for index in range(len(self)):
            yield self[index]

    def __hash__(self) -> int:
        # Invariant: If two vectors have the same value type, length, and value sequence, they will have the same hash.
        # This does not depend on the backend or vector subclass.
        if self._hash_value is None:
            self._hash_value = reduce(lambda a, b: (a << 1) ^ hash(b),
                                      self,
                                      (hash(self.value_type) << 1) ^ hash(len(self)))
        return self._hash_value

    def __eq__(self, other: Any) -> bool:
        # Invariant: If two vectors have the same value type, length, and value sequence, they will be equal.
        # This does not depend on the backend or vector subclass.
        return (isinstance(other, BackendVector) and
                self.value_type is other.value_type and
                len(self) == len(other) and
                self.eq(other).all())

    def __ne__(self, other: Any) -> bool:
        return not self == other


assert issubclass(BackendVector, ABC)


class Vector(Sequence[_FrontendValue]):
    _default_backend: 'Optional[VectorBackend]' = None

    @classmethod
    def get_default_backend(cls) -> 'VectorBackend':
        if cls._default_backend is None:
            raise ValueError("No default backend has been set.")
        return cls._default_backend

    @classmethod
    def set_default_backend(cls, backend: 'VectorBackend') -> None:
        cls._default_backend = backend

    @classmethod
    def clear_default_backend(cls) -> None:
        cls._default_backend = None

    @classmethod
    def default_backend_is_set(cls) -> bool:
        return cls._default_backend is not None

    @classmethod
    def new(cls, value_type: type[_FrontendValue], values: Iterable[_FrontendValue]) -> 'Vector[_FrontendValue]':
        if value_type is bool:
            return BoolVector(values)
        elif value_type is int:
            return IntVector(values)
        elif value_type is float:
            return FloatVector(values)
        else:
            return Vector(value_type, values)

    # @classmethod
    # def get_supported_dist_families(cls) -> frozenset[str]:
    #     return cls.get_default_backend().get_supported_dist_families()
    #
    # @classmethod
    # def dist_family_is_supported(cls, name: str) -> bool:
    #     return name in cls.get_supported_dist_families()
    #
    # @classmethod
    # def get_required_dist_family_params(cls, name: str) -> frozenset[str]:
    #     return
    #
    # @classmethod
    # def sample(cls, value_type: type[_FrontendValue], length: int, dist_family: str = 'uniform',
    #            **kwargs) -> 'Vector[_FrontendValue]':
    #     vector_type = cls.get_default_backend().get_vector_type(frontend_type=value_type)
    #     values = vector_type.sample(length, dist_family=dist_family, **kwargs)
    #     return cls.new(value_type, values)

    @classmethod
    def sample_uniform(cls, value_type: type[_FrontendValue], length: int, *, minimum: _FrontendValue = None,
                       maximum: _FrontendValue = None) -> 'Vector[_FrontendValue]':
        if minimum is None:
            minimum = 0
        if maximum is None:
            maximum = 1
        backend = cls.get_default_backend()
        backend_type = backend.get_default_backend_type(value_type)
        vector_type = backend.get_vector_type(backend_type=backend_type)
        to_backend_type = backend.get_call_conversion(value_type, backend_type)
        minimum = to_backend_type(minimum)
        maximum = to_backend_type(maximum)
        if not (minimum < maximum):
            raise ValueError("Minimum must be strictly less than maximum.")
        values = vector_type.sample_uniform(backend_type, length, minimum=minimum, maximum=maximum)
        return cls.new(value_type, values)

    @classmethod
    def sample_categorical(cls, value_type: type[_FrontendValue], length: int, *,
                           choices: Iterable[_FrontendValue] = None,
                           weights: Mapping[_FrontendValue, float] = None) -> 'Vector[_FrontendValue]':
        if choices is None and weights is None:
            raise ValueError("At least one of choices or weights must be specified.")
        # Note: We specifically do not require choices and weights to contain the same
        # entries. For choices that don't appear in the weights, a weight of 0.0 is assumed.
        # For weight keys that don't appear in choices, the weight key will be removed.
        if choices is not None:
            choices = set(choices)
        if weights is None:
            assert choices is not None
            weights = {choice: 1.0 for choice in choices}
        else:
            weights = dict(weights)
            if any(value < 0.0 for value in weights.values()):
                raise ValueError("Weights must be non-negative.")
            if choices is None:
                choices = weights.keys()
            else:
                weights = {key: value for key, value in weights.items() if key in choices}
                for key in choices:
                    if key not in weights:
                        weights[key] = 0.0
            if all(value == 0.0 for value in weights.values()):
                weights = {key: 1.0 for key in weights}
        assert choices == weights.keys()
        assert weights
        assert all(value >= 0.0 for value in weights.values())
        weights = {key: value for key, value in weights.items() if value > 0.0}
        total = sum(weights.values())
        assert total > 0.0
        if total != 1.0:
            for key, value in weights.items():
                weights[key] = value / total
        backend = cls.get_default_backend()
        backend_type = backend.get_default_backend_type(value_type)
        vector_type = backend.get_vector_type(backend_type=backend_type)
        to_backend_type = backend.get_call_conversion(value_type, backend_type)
        backend_weights = {}
        for key, value in weights.items():
            key = to_backend_type(key)
            backend_weights[key] = backend_weights.get(key, 0.0) + value
        values = vector_type.sample_categorical(backend_type, length, weights=backend_weights)
        return cls.new(value_type, values)

    @classmethod
    def sample_gaussian(cls, value_type: type[_FrontendValue], length: int, *, mean: _FrontendValue = 0,
                        stddev: _FrontendValue = 1) -> 'Vector[_FrontendValue]':
        # NOTE: We allow value_type to be other types besides float, so long as they are "float-like". One example of a
        # float-like type is decimal.Decimal.
        backend = cls.get_default_backend()
        backend_type = backend.get_default_backend_type(value_type)
        vector_type = backend.get_vector_type(backend_type=backend_type)
        to_backend_type = backend.get_call_conversion(value_type, backend_type)
        mean = to_backend_type(mean)
        stddev = to_backend_type(abs(stddev))
        values = vector_type.sample_gaussian(backend_type, length, mean=mean, stddev=stddev)
        return cls.new(value_type, values)

    def __init__(self, value_type: type[_FrontendValue],
                 values: Iterable[_FrontendValue] | BackendVector[_BackendValue]):
        if isinstance(values, BackendVector):
            backend_vector = values
        else:
            backend_vector = self.get_default_backend().new(values, frontend_type=value_type)
        self._value_type = value_type
        self._backend_vector = backend_vector

    @property
    def value_type(self) -> type[_FrontendValue]:
        return self._value_type

    def cast(self, value_type: type[_FrontendValue2]) -> 'Vector[_FrontendValue2]':
        backend = self._backend_vector.get_backend()
        backend_value_type = backend.get_default_backend_type(value_type)
        if backend.get_default_frontend_type(backend_value_type) is value_type:
            return self.new(value_type, self._backend_vector)
        return self.new(value_type, self._backend_vector.cast(backend_value_type))

    def all(self) -> bool:
        return self._backend_vector.all()

    def any(self) -> bool:
        return self._backend_vector.any()

    def count(self, value: _FrontendValue) -> int:
        backend = self._backend_vector.get_backend()
        conversion = backend.get_call_conversion(self._value_type, self._backend_vector.value_type)
        return self._backend_vector.count(conversion(value))

    def __len__(self) -> int:
        return len(self._backend_vector)

    def __str__(self) -> str:
        return repr(tuple(self))

    def __repr__(self) -> str:
        return type(self).__name__ + '(' + self.value_type.__name__ + ', ' + repr(self._backend_vector) + ')'

    def __iter__(self) -> Iterator[_FrontendValue]:
        backend = self._backend_vector.get_backend()
        conversion = backend.get_return_conversion(self._backend_vector.value_type)
        for backend_value in self._backend_vector:
            yield conversion(backend_value)

    @overload
    def __getitem__(self, index: int) -> '_FrontendValue': ...

    @overload
    def __getitem__(self, index: slice) -> 'Vector[_FrontendValue]': ...

    def __getitem__(self, index: int | slice) -> '_FrontendValue | FrontendVector[_FrontendValue]':
        if isinstance(index, int):
            if self._value_type is self._backend_vector.value_type:
                return self._backend_vector[index]
            backend = self._backend_vector.get_backend()
            conversion = backend.get_return_conversion(self._backend_vector.value_type, self._value_type)
            return conversion(self._backend_vector[index])
        else:
            return self.new(self._value_type, self._backend_vector[index])

    def __hash__(self) -> int:
        return (hash(self._value_type) << 1) ^ hash(self._backend_vector)

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Vector) and
                self.value_type is other.value_type and
                self._backend_vector == other._backend_vector)

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __reversed__(self) -> 'Vector[_FrontendValue]':
        return self.new(self._value_type, self._backend_vector.reversed())

    def __abs__(self) -> 'Vector[_FrontendValue]':
        return self.new(self._value_type, self._backend_vector.abs())

    def concat(self, *others: 'Vector[_FrontendValue]') -> 'Vector[_FrontendValue]':
        if (self._backend_vector.get_backend() is not other._backend_vector.get_backend() or
                self._backend_vector.value_type is not other._backend_vector.value_type):
            backend = self._backend_vector.get_backend()
            other = backend.new(other._backend_vector, backend_type=self._backend_vector.value_type)
        else:
            other = other._backend_vector
        return self.new(self._value_type, self._backend_vector.concat(other))

    def nonzero(self) -> 'Vector[bool]':
        return self.new(bool, self._backend_vector.nonzero())

    @optional_method
    def sum(self) -> _FrontendValue: ...

    @optional_method
    def mean(self) -> _FrontendValue: ...

    @optional_method
    def stddev(self, mean: _FrontendValue = None) -> _FrontendValue: ...

    def _elementwise_binary_op(self, value_type: type[_FrontendValue2],
                               op: Callable[[BackendVector[_BackendValue]], BackendVector[_BackendValue2]],
                               other: 'Vector[_FrontendValue]') -> 'Vector[_FrontendValue2]':
        # Validate length and type
        if len(self) != len(other):
            raise ValueError("Vector length mismatch.")
        if self._value_type != other._value_type:
            raise TypeError("Vector type mismatch.")
        # Convert backend vector to required type
        backend = self._backend_vector.get_backend()
        if other._backend_vector.get_backend() is not backend:
            other = backend.new(iter(other), backend_type=self._backend_vector.value_type)
        elif other._backend_vector.value_type is not self._backend_vector.value_type:
            other = other._backend_vector.cast(self._backend_vector.value_type)
        else:
            other = other._backend_vector
        # Perform operation & wrap result
        return self.new(value_type, op(other))

    def eq(self, other: 'Vector[_FrontendValue2]') -> 'Vector[bool]':
        return self._elementwise_binary_op(bool, self._backend_vector.eq, other)

    def ne(self, other: 'Vector[_FrontendValue2]') -> 'Vector[bool]':
        return self._elementwise_binary_op(bool, self._backend_vector.ne, other)

    def le(self, other: 'Vector[_FrontendValue2]') -> 'Vector[bool]':
        return self._elementwise_binary_op(bool, self._backend_vector.le, other)

    def ge(self, other: 'Vector[_FrontendValue2]') -> 'Vector[bool]':
        return self._elementwise_binary_op(bool, self._backend_vector.ge, other)

    def lt(self, other: 'Vector[_FrontendValue2]') -> 'Vector[bool]':
        return self._elementwise_binary_op(bool, self._backend_vector.lt, other)

    def gt(self, other: 'Vector[_FrontendValue2]') -> 'Vector[bool]':
        return self._elementwise_binary_op(bool, self._backend_vector.gt, other)


class BoolVector(Vector[bool]):
    _backend_vector: 'BoolBackendVector'
    
    def __init__(self, values: 'Iterable[bool] | BoolBackendVector'):
        super().__init__(bool, values)

    def nonzero(self) -> 'BoolVector':
        return self

    def __str__(self) -> str:
        return ''.join('1' if bit else '0' for bit in self)

    def __and__(self, other: 'BoolVector') -> 'BoolVector':
        return type(self)(self._backend_vector.and_(other._backend_vector))

    def __or__(self, other: 'BoolVector') -> 'BoolVector':
        return type(self)(self._backend_vector.or_(other._backend_vector))

    def __xor__(self, other: 'BoolVector') -> 'BoolVector':
        return type(self)(self._backend_vector.xor(other._backend_vector))

    def __invert__(self) -> 'BoolVector':
        return type(self)(self._backend_vector.not_())


class IntVector(Vector[int]):
    _backend_vector: 'IntBackendVector'

    def __init__(self, values: 'Iterable[int] | IntBackendVector'):
        super().__init__(int, values)

    def __abs__(self) -> 'IntVector':
        return type(self)(self._backend_vector.abs())

    def sign(self) -> 'IntVector':
        return type(self)(self._backend_vector.sign())

    def __neg__(self) -> 'IntVector':
        return type(self)(self._backend_vector.neg())

    def __add__(self, other: 'int | IntVector') -> 'IntVector':
        if isinstance(other, IntVector):
            other = other._backend_vector
        return type(self)(self._backend_vector.add(other))

    def __sub__(self, other: 'int | IntVector') -> 'IntVector':
        if isinstance(other, IntVector):
            other = other._backend_vector
        return type(self)(self._backend_vector.sub(other))

    def __mul__(self, other: 'int | IntVector') -> 'IntVector':
        if isinstance(other, IntVector):
            other = other._backend_vector
        return type(self)(self._backend_vector.mul(other))

    def __floordiv__(self, other: 'int | IntVector') -> 'IntVector':
        if isinstance(other, IntVector):
            other = other._backend_vector
        return type(self)(self._backend_vector.floordiv(other))

    def __mod__(self, other: 'int | IntVector') -> 'IntVector':
        if isinstance(other, IntVector):
            other = other._backend_vector
        return type(self)(self._backend_vector.mod(other))

    def __truediv__(self, other: 'int | IntVector') -> 'FloatVector':
        if isinstance(other, IntVector):
            other = other._backend_vector
        return self.new(float, self._backend_vector.truediv(other))

    def __pow__(self, power: 'int | IntVector', modulo=None) -> 'IntVector':
        assert modulo is None  # TODO
        if isinstance(power, IntVector):
            power = power._backend_vector
        return type(self)(self._backend_vector.pow(power))


class FloatVector(Vector[float]):
    _backend_vector: 'FloatBackendVector'

    def __init__(self, values: 'Iterable[float] | FloatBackendVector'):
        super().__init__(float, values)

    @Vector.sum.define
    def sum(self) -> float:
        return self._backend_vector.sum()

    @Vector.mean.define
    def mean(self) -> float:
        return self._backend_vector.mean()

    @Vector.stddev.define
    def stddev(self, mean: float = None) -> float:
        if mean is not None:
            mean = self.get_default_backend().get_call_conversion(float, self._backend_vector.value_type)(mean)
        return self._backend_vector.stddev(mean)

    def __abs__(self) -> 'FloatVector':
        return type(self)(self._backend_vector.abs())

    def sign(self) -> 'FloatVector':
        return type(self)(self._backend_vector.sign())

    def __neg__(self) -> 'FloatVector':
        return type(self)(self._backend_vector.neg())

    def __add__(self, other: 'float | FloatVector') -> 'FloatVector':
        if type(other) is FloatVector:
            other = other._backend_vector
        return type(self)(self._backend_vector.add(other))

    def __sub__(self, other: 'float | FloatVector') -> 'FloatVector':
        if type(other) is FloatVector:
            other = other._backend_vector
        return type(self)(self._backend_vector.sub(other))

    def __mul__(self, other: 'float | FloatVector') -> 'FloatVector':
        if type(other) is FloatVector:
            other = other._backend_vector
        return type(self)(self._backend_vector.mul(other))

    def __floordiv__(self, other: 'float | FloatVector') -> 'FloatVector':
        if type(other) is FloatVector:
            other = other._backend_vector
        return type(self)(self._backend_vector.floordiv(other))

    def __mod__(self, other: 'float | FloatVector') -> 'FloatVector':
        if type(other) is FloatVector:
            other = other._backend_vector
        return type(self)(self._backend_vector.mod(other))

    def __truediv__(self, other: 'float | FloatVector') -> 'FloatVector':
        if type(other) is FloatVector:
            other = other._backend_vector
        return type(self)(self._backend_vector.truediv(other))

    def __pow__(self, power: 'float | FloatVector', modulo=None) -> 'FloatVector':
        assert modulo is None  # TODO
        if type(power) is FloatVector:
            other = power._backend_vector
        return type(self)(self._backend_vector.pow(power))

    def exp(self) -> 'FloatVector':
        return type(self)(self._backend_vector.exp())

    def exp2(self) -> 'FloatVector':
        return type(self)(self._backend_vector.exp2())

    def log(self) -> 'FloatVector':
        return type(self)(self._backend_vector.log())

    def log2(self) -> 'FloatVector':
        return type(self)(self._backend_vector.log2())


@dataclass
class TypeConversionMapEntry(Generic[_ValueType, _ValueType2]):
    from_type: type[_ValueType]
    default: type[_ValueType2] = None
    conversions: dict[type[_ValueType2], Callable[[_ValueType], _ValueType2]] = field(default_factory=dict)


# noinspection PyTypeChecker
class VectorBackend:
    _call_conversions: dict[type[_FrontendValue], TypeConversionMapEntry[_FrontendValue, _BackendValue]]
    _return_conversions: dict[type[_BackendValue], TypeConversionMapEntry[_BackendValue, _FrontendValue]]
    _vector_types: 'dict[tuple[type[_BackendValue], bool], type[BackendVector[_BackendValue]]]'

    def __init__(self, name: str):
        self._name = name
        self._call_conversions = {}
        self._return_conversions = {}
        self._vector_types = {}

    @property
    def name(self) -> str:
        return self._name

    def activate(self) -> None:
        Vector.set_default_backend(self)

    def register_call_conversion(self, frontend_type: type[_FrontendValue], backend_type: type[_BackendValue],
                                 conversion: Callable[[_FrontendValue], _BackendValue] = None, *,
                                 default: bool = False) -> None:
        assert isinstance(frontend_type, type)
        assert isinstance(backend_type, type)
        assert conversion is None or callable(conversion)
        entry = self._call_conversions.get(frontend_type, None)
        if entry is None:
            entry = TypeConversionMapEntry(frontend_type)
        if backend_type in entry.conversions:
            raise TypeError(f"A conversion from frontend type {frontend_type.__name__} to backend type "
                            f"{backend_type.__name__} is already registered.")
        entry.conversions[backend_type] = conversion or backend_type
        if default:
            if entry.default is not None:
                raise TypeError(f"A default backend type is already mapped for frontend type {frontend_type.__name__}.")
            entry.default = backend_type
        LOGGER.info(f"Logging {backend_type.__module__}.{backend_type.__name__} as {'the default' if default else 'a'} "
                    f"backend type for {frontend_type.__module__}.{frontend_type.__name__} for {self._name} backend.")
        self._call_conversions[frontend_type] = entry

    def register_return_conversion(self, backend_type: type[_BackendValue], frontend_type: type[_FrontendValue],
                                   conversion: Callable[[_BackendValue], _FrontendValue] = None, *,
                                   default: bool = False) -> None:
        assert isinstance(frontend_type, type)
        assert isinstance(backend_type, type)
        assert conversion is None or callable(conversion)
        entry = self._return_conversions.get(backend_type, None)
        if entry is None:
            entry = TypeConversionMapEntry(backend_type)
        if frontend_type in entry.conversions:
            raise TypeError(f"A conversion from backend type {backend_type.__name__} to frontend type "
                            f"{frontend_type.__name__} is already registered.")
        entry.conversions[frontend_type] = conversion or frontend_type
        if default:
            if entry.default is not None:
                raise TypeError(f"A default frontend type is already mapped for backend type {backend_type.__name__}.")
            entry.default = frontend_type
        LOGGER.info(f"Logging {frontend_type.__module__}.{frontend_type.__name__} as "
                    f"{'the default' if default else 'a'} frontend type for "
                    f"{backend_type.__module__}.{backend_type.__name__} for {self._name} backend.")
        self._return_conversions[backend_type] = entry

    def register_dual_conversion(self, frontend_type: type[_FrontendValue], backend_type: type[_BackendValue],
                                 call_conversion: Callable[[_FrontendValue], _BackendValue] = None,
                                 return_conversion: Callable[[_BackendValue], _FrontendValue] = None, *,
                                 call_default: bool = False, return_default: bool = False) -> None:
        self.register_call_conversion(frontend_type, backend_type, call_conversion, default=call_default)
        self.register_return_conversion(backend_type, frontend_type, return_conversion, default=return_default)

    def get_call_conversion(self, frontend_type: type[_FrontendValue],
                            backend_type: type[_BackendValue] = None) -> Callable[[_FrontendValue], _BackendValue]:
        assert isinstance(frontend_type, type)
        assert backend_type is None or isinstance(backend_type, type), backend_type
        entry = self._call_conversions[frontend_type]
        if backend_type is None:
            backend_type = entry.default
            if backend_type is None:
                raise TypeError(f"No default type conversion has been set for frontend type {frontend_type.__name__}.")
        return entry.conversions[backend_type]

    def get_return_conversion(self, backend_type: type[_BackendValue],
                              frontend_type: type[_FrontendValue] = None) -> Callable[[_BackendValue], _FrontendValue]:
        assert frontend_type is None or isinstance(frontend_type, type)
        assert isinstance(backend_type, type)
        entry = self._return_conversions[backend_type]
        if frontend_type is None:
            frontend_type = entry.default
            if frontend_type is None:
                raise TypeError(f"No default type conversion has been set for backend type {backend_type.__name__}.")
        return entry.conversions[frontend_type]

    def get_default_backend_type(self, frontend_type: type[_FrontendValue]) -> Optional[type[_BackendValue]]:
        assert isinstance(frontend_type, type)
        entry = self._call_conversions.get(frontend_type, None)
        if entry is None:
            return None
        return entry.default

    def get_default_frontend_type(self, backend_type: type[_BackendValue]) -> Optional[type[_FrontendValue]]:
        assert isinstance(backend_type, type)
        entry = self._return_conversions.get(backend_type, None)
        if entry is None:
            return None
        return entry.default

    def _map_backend_type(self, frontend_type: type[_FrontendValue] | None,
                          backend_type: type[_BackendValue] | None) -> type[_BackendValue]:
        assert frontend_type is None or backend_type is None
        assert frontend_type is not None or backend_type is not None
        if backend_type is None:
            backend_type = self.get_default_backend_type(frontend_type)
            if backend_type is None:
                raise TypeError(f"No default backend type has been registered for frontend type "
                                f"{frontend_type.__name__}.")
        assert isinstance(backend_type, type)
        return backend_type

    def _map_frontend_type(self, frontend_type: type[_FrontendValue] | None,
                           backend_type: type[_BackendValue] | None) -> type[_FrontendValue]:
        assert frontend_type is None or backend_type is None
        assert frontend_type is not None or backend_type is not None
        if frontend_type is None:
            frontend_type = self.get_default_frontend_type(backend_type)
            if frontend_type is None:
                raise TypeError(f"No default frontend type has been registered for backend type "
                                f"{backend_type.__name__}.")
        assert isinstance(frontend_type, type)
        return frontend_type

    @overload
    def get_vector_type(self, *, frontend_type: type[_FrontendValue],
                        sparse: bool = None, required: bool = True) -> 'Optional[type[BackendVector[_BackendValue]]]':
        ...

    @overload
    def get_vector_type(self, *, backend_type: type[_BackendValue],
                        sparse: bool = None, required: bool = True) -> 'Optional[type[BackendVector[_BackendValue]]]':
        ...

    def get_vector_type(self, *, frontend_type: type[_FrontendValue] = None, backend_type: type[_BackendValue] = None,
                        sparse: bool = None, required: bool = True) -> 'Optional[type[BackendVector[_BackendValue]]]':
        backend_type = self._map_backend_type(frontend_type, backend_type)
        if sparse is None:
            sparse_flags = (False, True)
        else:
            sparse_flags = (bool(sparse),)
        for sparse in sparse_flags:
            key = (backend_type, sparse)
            if key in self._vector_types:
                return self._vector_types[key]
        if required:
            raise ValueError(f"No vector type has been registered for backend type {backend_type.__name__}.")
        return None

    @overload
    def new(self, values: Iterable[_FrontendValue], *, frontend_type: type[_FrontendValue],
            sparse: bool = None) -> 'BackendVector[_BackendValue]':
        ...

    @overload
    def new(self, values: Iterable[_BackendValue], *, backend_type: type[_BackendValue],
            sparse: bool = None) -> 'BackendVector[_BackendValue]':
        ...

    def new(self, values: Iterable[_FrontendValue | _BackendValue], *, backend_type: type[_BackendValue] = None,
            frontend_type: type[_FrontendValue] = None, sparse: bool = None) -> 'BackendVector[_BackendValue]':
        backend_type = self._map_backend_type(frontend_type, backend_type)
        vector_type = self.get_vector_type(backend_type=backend_type, sparse=sparse)
        return vector_type.new(backend_type, values)

    def register(self, vector_class: type[BackendVector]) -> None:
        sparse = vector_class.is_sparse()
        backend_type: type[_BackendValue]
        for backend_type in vector_class.get_supported_value_types():
            key = (backend_type, sparse)
            LOGGER.info(f"Registering {vector_class.__module__}.{vector_class.__name__} as the "
                        f"{'sparse' if sparse else 'dense'} vector class for backend value type "
                        f"{backend_type.__module__}.{backend_type.__name__}.")
            if key in self._vector_types:
                raise TypeError(f"A {'sparse' if sparse else 'dense'} {backend_type.__name__} vector class has already "
                                f"been registered for this backend.")
            self._vector_types[backend_type, sparse] = vector_class

    def validate(self) -> None:
        for frontend_type, backend_interface in [(bool, BoolBackendVector),
                                                 (int, IntBackendVector),
                                                 (float, FloatBackendVector)]:
            backend_type = self.get_default_backend_type(frontend_type)
            if backend_type is not None:
                for sparse in (False, True):
                    vector_type = self.get_vector_type(backend_type=backend_type, sparse=sparse, required=False)
                    if vector_type is not None:
                        if not issubclass(vector_type, backend_interface):
                            raise TypeError(vector_type, backend_interface)


class BoolBackendVector(BackendVector[_BackendBool], ABC):

    @abstractmethod
    def not_(self) -> 'BoolBackendVector[_BackendBool]': ...

    @abstractmethod
    def and_(self, other: 'BoolBackendVector[_BackendBool]') -> 'BoolBackendVector[_BackendBool]': ...

    @abstractmethod
    def or_(self, other: 'BoolBackendVector[_BackendBool]') -> 'BoolBackendVector[_BackendBool]': ...

    @abstractmethod
    def xor(self, other: 'BoolBackendVector[_BackendBool]') -> 'BoolBackendVector[_BackendBool]': ...


class IntBackendVector(BackendVector[_BackendValue], ABC):

    @abstractmethod
    def abs(self) -> 'IntBackendVector[_BackendValue]': ...

    @abstractmethod
    def sign(self) -> 'IntBackendVector[_BackendValue]': ...

    @abstractmethod
    def neg(self) -> 'IntBackendVector[_BackendValue]': ...

    @abstractmethod
    def add(self, other: 'int | IntBackendVector[_BackendValue]') -> 'IntBackendVector[_BackendValue]': ...

    @abstractmethod
    def sub(self, other: 'int | IntBackendVector[_BackendValue]') -> 'IntBackendVector[_BackendValue]': ...

    @abstractmethod
    def mul(self, other: 'int | IntBackendVector[_BackendValue]') -> 'IntBackendVector[_BackendValue]': ...

    @optional_method
    def floordiv(self, other: 'int | IntBackendVector[_BackendValue]') -> 'IntBackendVector[_BackendValue]': ...

    @optional_method
    def mod(self, other: 'int | IntBackendVector[_BackendValue]') -> 'IntBackendVector[_BackendValue]': ...

    @optional_method
    def truediv(self, other: 'int | IntBackendVector[_BackendValue]') -> 'FloatBackendVector[_BackendValue2]': ...

    @optional_method
    def pow(self, other: 'int | IntBackendVector[_BackendValue]') -> 'IntBackendVector[_BackendValue]': ...


class FloatBackendVector(BackendVector[_BackendValue], ABC):

    @abstractmethod
    def sum(self) -> _BackendValue: ...

    @abstractmethod
    def mean(self) -> _BackendValue: ...

    @abstractmethod
    def stddev(self, mean: _BackendValue = None) -> _BackendValue: ...

    @abstractmethod
    def abs(self) -> 'FloatBackendVector[_BackendValue]': ...

    @abstractmethod
    def sign(self) -> 'FloatBackendVector[_BackendValue]': ...

    @abstractmethod
    def neg(self) -> 'FloatBackendVector[_BackendValue]': ...

    @abstractmethod
    def add(self, other: 'float | FloatBackendVector[_BackendValue]') -> 'FloatBackendVector[_BackendValue]': ...

    @abstractmethod
    def sub(self, other: 'float | FloatBackendVector[_BackendValue]') -> 'FloatBackendVector[_BackendValue]': ...

    @abstractmethod
    def mul(self, other: 'float | FloatBackendVector[_BackendValue]') -> 'FloatBackendVector[_BackendValue]': ...

    @optional_method
    def floordiv(self, other: 'float | FloatBackendVector[_BackendValue]') -> 'FloatBackendVector[_BackendValue]': ...

    @optional_method
    def mod(self, other: 'float | FloatBackendVector[_BackendValue]') -> 'FloatBackendVector[_BackendValue]': ...

    @optional_method
    def truediv(self, other: 'float | FloatBackendVector[_BackendValue]') -> 'FloatBackendVector[_BackendValue]': ...

    @optional_method
    def pow(self, other: 'float | FloatBackendVector[_BackendValue]') -> 'FloatBackendVector[_BackendValue]': ...

    @optional_method
    def exp(self) -> 'FloatBackendVector[_BackendValue]': ...

    @optional_method
    def exp2(self) -> 'FloatBackendVector[_BackendValue]': ...

    @optional_method
    def log(self) -> 'FloatBackendVector[_BackendValue]': ...

    @optional_method
    def log2(self) -> 'FloatBackendVector[_BackendValue]': ...
