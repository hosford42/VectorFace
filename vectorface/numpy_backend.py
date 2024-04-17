from abc import ABC
from typing import Callable, TypeVar, Hashable, overload, Iterable

# If numpy isn't installed, this will (intentionally) raise an ImportError
import numpy as np

from .base import BackendVector, VectorBackend, BoolBackendVector, IntBackendVector, FloatBackendVector

# Sometimes numpy isn't uninstalled properly and what's left is an empty
# folder. It's unusable, but still imports without error. This ensures we
# don't take it for granted that the module is usable.
try:
    np.ndarray
except AttributeError:
    raise ImportError('The numpy module failed to uninstall properly.')

__all__ = [
    'NUMPY_BACKEND',
    'NumpyVectorBase',
    'NumpyBoolVector',
    'NumpyIntVector',
    'NumpyInt8Vector',
    'NumpyInt16Vector',
    'NumpyInt32Vector',
    'NumpyInt64Vector',
    'NumpyFloatVector',
    'NumpyFloat16Vector',
    'NumpyFloat32Vector',
    'NumpyFloat64Vector',
]

# TODO:
#   * Check how many bits python's builtin float takes. Adjust type definitions below based on that.

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


NUMPY_BACKEND = VectorBackend('numpy')
NUMPY_BACKEND.register_dual_conversion(bool, np.bool_, call_default=True, return_default=True)
NUMPY_BACKEND.register_dual_conversion(int, np.int8, return_default=True)
NUMPY_BACKEND.register_dual_conversion(int, np.int16, return_default=True)
NUMPY_BACKEND.register_dual_conversion(int, np.int32, call_default=True, return_default=True)
NUMPY_BACKEND.register_dual_conversion(int, np.int64, return_default=True)
NUMPY_BACKEND.register_dual_conversion(float, np.float16, return_default=True)
NUMPY_BACKEND.register_dual_conversion(float, np.float32, return_default=True)
NUMPY_BACKEND.register_dual_conversion(float, np.float64, call_default=True, return_default=True)


_Value = TypeVar('_Value', bound=Hashable)


class NumpyVectorBase(BackendVector[_Value], ABC):
    _dtype: type[_Value]

    @classmethod
    def new(cls, value_type: type[_BackendValue], values: Iterable[_ValueType]) -> 'NumpyVectorBase[_BackendValue]':
        assert value_type is cls._dtype
        return cls(values)

    @classmethod
    def sample_categorical(cls, value_type: type[_BackendValue], length: int, *,
                           weights: dict[_BackendValue, float]) -> 'BackendVector[_BackendValue]':
        assert length >= 0
        if len(weights) == 1:
            for key in weights:
                return cls.new(value_type, np.repeat(np.array([key], dtype=value_type), length))
        keys = list(weights)
        values = [weights[key] for key in keys]
        return cls.new(value_type, np.random.choice(keys, length, replace=True, p=values))

    @classmethod
    def sample_gaussian(cls, value_type: type[_BackendValue], length: int, *, mean: _BackendValue,
                        stddev: _BackendValue) -> 'BackendVector[_BackendValue]':
        raise NotImplementedError(f"Operation not supported for value type {value_type.__name__}.")

    @classmethod
    def get_backend(cls) -> 'VectorBackend':
        return NUMPY_BACKEND

    @classmethod
    def get_supported_value_types(cls) -> tuple[type[_Value], ...]:
        dtype = getattr(cls, '_dtype', None)
        if dtype is None:
            return ()
        return cls._dtype,

    @classmethod
    def is_sparse(cls) -> bool:
        # TODO: Does numpy support sparse vectors? If so, then so should we.
        return False

    def __init__(self, values: Iterable[_ValueType]):
        # TODO: If values is already an ndarray, and is immutable, reuse it.
        self._values: np.ndarray = np.array(values, self._dtype, copy=True)

    @property
    def value_type(self) -> type[_BackendValue]:
        return self._dtype

    def cast(self, value_type: type[_BackendValue2]) -> 'BackendVector[_BackendValue2]':
        return self.get_backend().new(self._values.astype(value_type), backend_type=value_type)

    def all(self) -> bool:
        return bool(np.all(self._values))

    def any(self) -> bool:
        return bool(np.any(self._values))

    def count(self, value: _ValueType) -> int:
        return int(np.count_nonzero(self._values == value))

    def __len__(self) -> int:
        return len(self._values)

    @overload
    def __getitem__(self, index: int) -> '_BackendValue': ...

    @overload
    def __getitem__(self, index: slice) -> 'BackendVector[_BackendValue]': ...

    def __getitem__(self, index: int | slice) -> '_BackendValue | Vector[_BackendValue]':
        if isinstance(index, int):
            return self._values[index]
        else:
            return self.new(self._dtype, self._values[index])

    def reversed(self) -> 'BackendVector[_BackendValue]':
        return type(self)(self._values[::-1])

    def concat(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[_BackendValue]':
        if not isinstance(other, NumpyVectorBase):
            other = self.new(self._dtype, other)
        return self.new(self._dtype, np.concatenate([self._values, other._values], axis=0))

    def eq(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[np.bool_]':
        if not isinstance(other, NumpyVectorBase):
            other = self.new(self._dtype, other)
        return self.get_backend().new(np.equal(self._values, other._values), backend_type=np.bool_)

    def ne(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[np.bool_]':
        if not isinstance(other, NumpyVectorBase):
            other = self.new(self._dtype, other)
        return self.get_backend().new(np.not_equal(self._values, other._values), backend_type=np.bool_)

    @BackendVector.le.define
    def le(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[np.bool_]':
        if not isinstance(other, NumpyVectorBase):
            other = self.new(self._dtype, other)
        return self.get_backend().new(np.less_equal(self._values, other._values), backend_type=np.bool_)

    @BackendVector.ge.define
    def ge(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[np.bool_]':
        if not isinstance(other, NumpyVectorBase):
            other = self.new(self._dtype, other)
        return self.get_backend().new(np.greater_equal(self._values, other._values), backend_type=np.bool_)

    @BackendVector.lt.define
    def lt(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[np.bool_]':
        if not isinstance(other, NumpyVectorBase):
            other = self.new(self._dtype, other)
        return self.get_backend().new(np.less(self._values, other._values), backend_type=np.bool_)

    @BackendVector.gt.define
    def gt(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[np.bool_]':
        if not isinstance(other, NumpyVectorBase):
            other = self.new(self._dtype, other)
        return self.get_backend().new(np.greater(self._values, other._values), backend_type=np.bool_)

    @BackendVector.abs.define
    def abs(self) -> 'BackendVector[_BackendValue]':
        return self.new(self._dtype, np.abs(self._values))


class NumpyBoolVector(NumpyVectorBase[np.bool_], BoolBackendVector):
    _dtype = np.bool_

    @classmethod
    def sample_uniform(cls, value_type: type[np.bool_], length: int, *, minimum: bool = False,
                       maximum: bool = True) -> 'NumpyBoolVector':
        assert value_type is np.bool_
        assert length >= 0
        if minimum != 0 or maximum != 1:
            raise ValueError("Unsupported minimum or maximum value for boolean vector.")
        return cls.new(value_type, np.random.randint(2, size=length, dtype=np.bool_))

    @classmethod
    def sample_categorical(cls, value_type: type[np.bool_], length: int, *,
                           weights: dict[np.bool_, float]) -> 'NumpyBoolVector':
        assert value_type is np.bool_
        assert length >= 0
        if len(weights) == 1:
            for key in weights:
                return cls.new(value_type, np.repeat([key], length))
        elif len(weights) == 2 and min(weights.values()) == max(weights.values()):
            return cls.sample_uniform(value_type, length)
        return super().sample_categorical(value_type, length, weights=weights)

    def and_(self, other: 'NumpyBoolVector') -> 'NumpyBoolVector':
        bits = np.bitwise_and(self._values, other._values)
        # Make sure the bit array isn't writable, so it can be used by the constructor
        bits.flags.writeable = False
        return type(self)(bits)

    def or_(self, other: 'NumpyBoolVector') -> 'NumpyBoolVector':
        bits = np.bitwise_or(self._values, other._values)
        # Make sure the bit array isn't writable, so it can be used by the constructor
        bits.flags.writeable = False
        return type(self)(bits)

    def xor(self, other: 'NumpyBoolVector') -> 'NumpyBoolVector':
        bits = np.bitwise_xor(self._values, other._values)
        # Make sure the bit array isn't writable, so it can be used by the constructor
        bits.flags.writeable = False
        return type(self)(bits)

    def not_(self) -> 'NumpyBoolVector':
        bits = np.bitwise_not(self._values)
        # Make sure the bit array isn't writable, so it can be used by the constructor
        bits.flags.writeable = False
        return type(self)(bits)


class NumpyIntVector(NumpyVectorBase[_BackendValue], IntBackendVector):

    @classmethod
    def sample_uniform(cls, value_type: type[_BackendValue], length: int, *, minimum: _BackendValue,
                       maximum: _BackendValue) -> 'NumpyIntVector[_BackendValue]':
        assert np.issubdtype(value_type, np.int_)
        assert length >= 0
        assert isinstance(minimum, value_type)
        assert isinstance(maximum, value_type)
        assert minimum < maximum
        return cls.new(value_type, np.random.randint(minimum, maximum + 1, size=length, dtype=value_type))

    def abs(self) -> 'NumpyIntVector[_BackendValue]':
        return type(self)(np.abs(self._values))

    def sign(self) -> 'NumpyIntVector[_BackendValue]':
        return type(self)(np.sign(self._values))

    def neg(self) -> 'NumpyIntVector[_BackendValue]':
        return type(self)(-self._values)

    def add(self, other: 'int | NumpyIntVector[_BackendValue]') -> 'NumpyIntVector[_BackendValue]':
        if isinstance(other, NumpyIntVector):
            other = other._values
        return type(self)(self._values + other)

    def sub(self, other: 'int | NumpyIntVector[_BackendValue]') -> 'NumpyIntVector[_BackendValue]':
        if isinstance(other, NumpyIntVector):
            other = other._values
        return type(self)(self._values - other)

    def mul(self, other: 'int | NumpyIntVector[_BackendValue]') -> 'NumpyIntVector[_BackendValue]':
        if isinstance(other, NumpyIntVector):
            other = other._values
        return type(self)(self._values * other)

    @IntBackendVector.floordiv.define
    def floordiv(self, other: 'int | NumpyIntVector[_BackendValue]') -> 'NumpyIntVector[_BackendValue]':
        if isinstance(other, NumpyIntVector):
            other = other._values
        return type(self)(self._values // other)

    @IntBackendVector.mod.define
    def mod(self, other: 'int | IntBackendVector[_BackendValue]') -> 'IntBackendVector[_BackendValue]':
        if isinstance(other, NumpyIntVector):
            other = other._values
        return type(self)(self._values % other)

    @IntBackendVector.truediv.define
    def truediv(self, other: 'int | NumpyIntVector[_BackendValue]') -> 'NumpyFloatVector[_BackendValue2]':
        if isinstance(other, NumpyIntVector):
            other = other._values
        # noinspection PyTypeChecker
        return self.get_backend().new(self._values / other, frontend_type=float)


class NumpyInt8Vector(NumpyIntVector[np.int8]):
    _dtype = np.int8


class NumpyInt16Vector(NumpyIntVector[np.int16]):
    _dtype = np.int16


class NumpyInt32Vector(NumpyIntVector[np.int32]):
    _dtype = np.int32


class NumpyInt64Vector(NumpyIntVector[np.int64]):
    _dtype = np.int64


class NumpyFloatVector(NumpyVectorBase[_Value], FloatBackendVector):

    @classmethod
    def sample_uniform(cls, value_type: type[_BackendValue], length: int, *, minimum: _BackendValue,
                       maximum: _BackendValue) -> 'NumpyFloatVector[_BackendValue]':
        assert np.issubdtype(value_type, np.float_)
        assert length >= 0
        assert isinstance(minimum, value_type)
        assert isinstance(maximum, value_type)
        assert minimum < maximum
        return cls.new(value_type, np.random.uniform(minimum, maximum, size=length).astype(value_type))

    @classmethod
    def sample_gaussian(cls, value_type: type[_BackendValue], length: int, *, mean: _BackendValue,
                        stddev: _BackendValue) -> 'BackendVector[_BackendValue]':
        assert np.issubdtype(value_type, np.float_)
        assert length >= 0
        return cls.new(value_type, np.random.normal(mean, stddev, size=length).astype(value_type))

    def sum(self) -> _BackendValue:
        return np.sum(self._values)

    def mean(self) -> _BackendValue:
        return np.mean(self._values)

    def stddev(self, mean: _BackendValue = None) -> _BackendValue:
        if mean is None:
            return np.std(self._values)
        return np.sqrt(np.square(self._values - mean) / len(self._values))

    def abs(self) -> 'NumpyFloatVector[_BackendValue]':
        return type(self)(np.abs(self._values))

    def sign(self) -> 'NumpyFloatVector[_BackendValue]':
        return type(self)(np.sign(self._values))

    def neg(self) -> 'NumpyFloatVector[_BackendValue]':
        return type(self)(-self._values)

    def add(self, other: 'float | NumpyVectorBase[_BackendValue]') -> 'NumpyFloatVector[_BackendValue]':
        if isinstance(other, NumpyVectorBase):
            other = other._values
        return type(self)(self._values + other)

    def sub(self, other: 'float | NumpyVectorBase[_BackendValue]') -> 'NumpyFloatVector[_BackendValue]':
        if isinstance(other, NumpyVectorBase):
            other = other._values
        return type(self)(self._values - other)

    def mul(self, other: 'float | NumpyVectorBase[_BackendValue]') -> 'NumpyFloatVector[_BackendValue]':
        if isinstance(other, NumpyVectorBase):
            other = other._values
        return type(self)(self._values * other)

    @FloatBackendVector.truediv.define
    def truediv(self, other: 'float | NumpyVectorBase[_BackendValue]') -> 'NumpyFloatVector[_BackendValue]':
        if isinstance(other, NumpyVectorBase):
            other = other._values
        return type(self)(self._values / other)

    @FloatBackendVector.floordiv.define
    def floordiv(self, other: 'float | NumpyVectorBase[_BackendValue]') -> 'NumpyFloatVector[_BackendValue]':
        if isinstance(other, NumpyVectorBase):
            other = other._values
        return type(self)(self._values // other)

    @FloatBackendVector.mod.define
    def mod(self, other: 'float | NumpyVectorBase[_BackendValue]') -> 'NumpyFloatVector[_BackendValue]':
        if isinstance(other, NumpyVectorBase):
            other = other._values
        return type(self)(self._values % other)

    @FloatBackendVector.truediv.define
    def truediv(self, other: 'float | NumpyVectorBase[_BackendValue]') -> 'NumpyFloatVector[_BackendValue]':
        if isinstance(other, NumpyVectorBase):
            other = other._values
        # noinspection PyTypeChecker
        return self.get_backend().new(self._values / other, frontend_type=float)


class NumpyFloat16Vector(NumpyFloatVector[np.float16]):
    _dtype = np.float16


class NumpyFloat32Vector(NumpyFloatVector[np.float32]):
    _dtype = np.float32


class NumpyFloat64Vector(NumpyFloatVector[np.float64]):
    _dtype = np.float64


NUMPY_BACKEND.validate()
