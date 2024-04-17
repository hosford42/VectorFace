import random
from typing import Callable, TypeVar, Hashable, overload, Iterable, Iterator

from .base import BackendVector, VectorBackend, BoolBackendVector, IntBackendVector, FloatBackendVector

__all__ = [
    'PYTHON_BACKEND',
    'PythonObjectVector',
    'PythonBoolVector',
    'PythonIntVector',
    'PythonFloatVector',
]

_BackendValue = TypeVar('_BackendValue', bound=Hashable)
_BackendValue2 = TypeVar('_BackendValue2', bound=Hashable)
_BackendValue3 = TypeVar('_BackendValue3', bound=Hashable)
_FrontendValue = TypeVar('_FrontendValue', bound=Hashable)
_FrontendValue2 = TypeVar('_FrontendValue2', bound=Hashable)
_FrontendValue3 = TypeVar('_FrontendValue3', bound=Hashable)


_Method = TypeVar('_Method', bound=Callable)


PYTHON_BACKEND = VectorBackend('numpy')
PYTHON_BACKEND.register_dual_conversion(bool, bool, call_default=True, return_default=True)
PYTHON_BACKEND.register_dual_conversion(int, int, call_default=True, return_default=True)
PYTHON_BACKEND.register_dual_conversion(float, float, call_default=True, return_default=True)
PYTHON_BACKEND.register_dual_conversion(object, object, call_default=True, return_default=True)


_Value = TypeVar('_Value', bound=Hashable)


class PythonObjectVector(BackendVector[_BackendValue]):
    _dtype: type[_BackendValue] = object

    @classmethod
    def new(cls, value_type: type[_BackendValue], 
            values: Iterable[_BackendValue]) -> 'PythonObjectVector[_BackendValue]':
        assert value_type is cls._dtype
        return cls(values)

    @classmethod
    def sample_uniform(cls, value_type: type[_BackendValue], length: int, *, minimum: _BackendValue,
                       maximum: _BackendValue) -> 'BackendVector[_BackendValue]':
        raise NotImplementedError(f"Operation not supported for value type {value_type.__name__}.")

    @classmethod
    def sample_categorical(cls, value_type: type[_BackendValue], length: int, *,
                           weights: dict[_BackendValue, float]) -> 'BackendVector[_BackendValue]':
        assert length >= 0
        if len(weights) == 1:
            for key in weights:
                return cls.new(value_type, [key] * length)
        values = []
        while len(values) < length:
            selector = random.random()
            for key, weight in weights.items():
                selector -= weight
                if selector <= 0.0:
                    values.append(key)
                    break
        return cls.new(value_type, values)

    @classmethod
    def sample_gaussian(cls, value_type: type[_BackendValue], length: int, *, mean: _BackendValue,
                        stddev: _BackendValue) -> 'BackendVector[_BackendValue]':
        raise NotImplementedError(f"Operation not supported for value type {value_type.__name__}.")

    @classmethod
    def get_backend(cls) -> 'VectorBackend':
        return PYTHON_BACKEND

    @classmethod
    def get_supported_value_types(cls) -> tuple[type[_BackendValue], ...]:
        dtype = getattr(cls, '_dtype', None)
        if dtype is None:
            return ()
        return cls._dtype,

    @classmethod
    def is_sparse(cls) -> bool:
        # TODO: Support sparse vectors.
        return False

    def __init__(self, values: Iterable[_BackendValue]):
        self._values: tuple[_BackendValue, ...] = tuple(values)

    @property
    def value_type(self) -> type[_BackendValue]:
        return self._dtype

    def cast(self, value_type: type[_BackendValue2]) -> 'BackendVector[_BackendValue2]':
        return self.get_backend().new((value_type(value) for value in self._values), backend_type=value_type)

    def all(self) -> bool:
        return all(self._values)

    def any(self) -> bool:
        return any(self._values)

    def count(self, value: _Value) -> int:
        return self._values.count(value)

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
        return type(self)(reversed(self._values))

    def concat(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[_BackendValue]':
        if not isinstance(other, PythonObjectVector):
            other = self.new(self._dtype, other)
        return self.new(self._dtype, self._values + other._values)

    def eq(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[bool]':
        if not isinstance(other, PythonObjectVector):
            other = self.new(self._dtype, other)
        return self.get_backend().new((mine == yours for mine, yours in zip(self._values, other._values)),
                                      backend_type=bool)

    def ne(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[bool]':
        if not isinstance(other, PythonObjectVector):
            other = self.new(self._dtype, other)
        return self.get_backend().new((mine != yours for mine, yours in zip(self._values, other._values)),
                                      backend_type=bool)

    @BackendVector.le.define
    def le(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[bool]':
        if not isinstance(other, PythonObjectVector):
            other = self.new(self._dtype, other)
        return self.get_backend().new((mine <= yours for mine, yours in zip(self._values, other._values)),
                                      backend_type=bool)

    @BackendVector.ge.define
    def ge(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[bool]':
        if not isinstance(other, PythonObjectVector):
            other = self.new(self._dtype, other)
        return self.get_backend().new((mine >= yours for mine, yours in zip(self._values, other._values)),
                                      backend_type=bool)

    @BackendVector.lt.define
    def lt(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[bool]':
        if not isinstance(other, PythonObjectVector):
            other = self.new(self._dtype, other)
        return self.get_backend().new((mine < yours for mine, yours in zip(self._values, other._values)),
                                      backend_type=bool)

    @BackendVector.gt.define
    def gt(self, other: 'BackendVector[_BackendValue]') -> 'BackendVector[bool]':
        if not isinstance(other, PythonObjectVector):
            other = self.new(self._dtype, other)
        return self.get_backend().new((mine > yours for mine, yours in zip(self._values, other._values)),
                                      backend_type=bool)

    @BackendVector.abs.define
    def abs(self) -> 'BackendVector[_BackendValue]':
        return self.new(self._dtype, (abs(value) for value in self._values))


class PythonBoolVector(BoolBackendVector, BackendVector[bool]):
    _dtype = bool

    @classmethod
    def new(cls, value_type: type[bool], values: Iterable[bool]) -> 'PythonBoolVector':
        assert value_type is bool
        return cls(values)

    @classmethod
    def sample_uniform(cls, value_type: type[bool], length: int, *, minimum: bool = False,
                       maximum: bool = True) -> 'PythonBoolVector':
        assert value_type is bool
        assert length >= 0
        if not (minimum is False and maximum is True):
            raise ValueError("Unsupported minimum or maximum value for boolean vector.")
        return cls(random.randrange(0, 1 << length), length=length)

    @classmethod
    def sample_categorical(cls, value_type: type[bool], length: int, *,
                           weights: dict[bool, float]) -> 'PythonBoolVector':
        assert value_type is bool
        assert length >= 0
        assert weights.keys() <= {False, True}
        if len(weights) == 1:
            for key in weights:
                return cls.new(value_type, [key] * length)
        elif not weights or weights[False] == weights[True]:
            return cls.sample_uniform(value_type, length)
        elif not weights[False]:
            return cls.new(value_type, [True] * length)
        elif not weights[True]:
            return cls.new(value_type, [False] * length)
        else:
            true_prob = weights[True]
            values = []
            while len(values) < length:
                values.append(random.random() <= true_prob)
            return cls.new(value_type, values)

    @classmethod
    def sample_gaussian(cls, value_type: type[bool], length: int, *, mean: bool,
                        stddev: bool) -> 'PythonBoolVector':
        raise NotImplementedError(f"Operation not supported for value type bool.")

    @classmethod
    def get_backend(cls) -> 'VectorBackend':
        return PYTHON_BACKEND

    @classmethod
    def get_supported_value_types(cls) -> tuple[type[bool], ...]:
        return bool,

    @classmethod
    def is_sparse(cls) -> bool:
        # TODO: Support sparse vectors.
        return False

    def __init__(self, values: int | Iterable[bool], *, length: int = None):
        self._bits: int
        self._length: int
        if isinstance(values, int):
            assert length is not None
            self._bits = values
            self._length = length
        elif isinstance(values, PythonBoolVector):
            self._bits = values._bits
            self._length = values._length
        else:
            values = list(values)
            values.reverse()
            bits = 0
            count = 0
            for bit in values:
                bits <<= 1
                bits |= bool(bit)
                count += 1
            self._bits = bits
            self._length = count
        assert length is None or self._length == length
        self._bits &= (1 << self._length) - 1

    @property
    def value_type(self) -> type[bool]:
        return bool

    def cast(self, value_type: type[_BackendValue2]) -> 'BackendVector[_BackendValue2]':
        return self.get_backend().new((value_type(value) for value in self), backend_type=value_type)

    def all(self) -> bool:
        return self._bits == (1 << self._length) - 1

    def any(self) -> bool:
        return bool(self._bits)

    def count(self, value: bool) -> int:
        if value:
            return self._bits.bit_count()
        else:
            return self._length - self._bits.bit_count()

    def __len__(self) -> int:
        return self._length

    @overload
    def __getitem__(self, index: int) -> 'bool': ...

    @overload
    def __getitem__(self, index: slice) -> 'BoolBackendVector': ...

    def __getitem__(self, index: int | slice) -> 'bool | BoolBackendVector':
        if type(index) is int:
            if 0 <= index < self._length:
                return bool((self._bits >> index) % 2)
            elif self._length <= index < 0:
                return bool((self._bits >> (self._length + index)) % 2)
            else:
                raise IndexError(index)
        elif type(index) is slice:
            if index.step == 1:
                start = index.start
                stop = index.stop
                length = stop - start
                if length > 0:
                    return type(self)((self._bits >> start) % (1 << (stop - start)), length=length)
                else:
                    return type(self)(0, length=0)
            return self.new(bool, (self[i] for i in range(*index.indices(self._length))))
        else:
            return self.__getitem__(int(index))

    def __iter__(self) -> Iterator[bool]:
        bits = self._bits
        for _ in range(self._length):
            yield bool(bits % 2)
            bits >>= 1

    def reversed(self) -> 'PythonBoolVector':
        return PythonBoolVector(reversed(self))

    def concat(self, other: 'BackendVector[_BackendValue]') -> 'PythonBoolVector':
        if not isinstance(other, PythonBoolVector):
            other = self.new(bool, other)
        return PythonBoolVector((other._bits << self._length) + self._bits & ((1 << self._length) - 1),
                                length=self._length + other._length)

    def eq(self, other: 'BackendVector[_BackendValue]') -> 'PythonBoolVector':
        if not isinstance(other, PythonObjectVector):
            other = self.new(bool, other)
        return PythonBoolVector(~(self._bits ^ other._bits), length=self._length)

    def ne(self, other: 'BackendVector[_BackendValue]') -> 'PythonBoolVector':
        if not isinstance(other, PythonObjectVector):
            other = self.new(bool, other)
        return PythonBoolVector(self._bits ^ other._bits, length=self._length)

    @BackendVector.le.define
    def le(self, other: 'BackendVector[_BackendValue]') -> 'PythonBoolVector':
        if not isinstance(other, PythonObjectVector):
            other = self.new(bool, other)
        return PythonBoolVector(~self._bits | other._bits, length=self._length)

    @BackendVector.ge.define
    def ge(self, other: 'BackendVector[_BackendValue]') -> 'PythonBoolVector':
        if not isinstance(other, PythonObjectVector):
            other = self.new(bool, other)
        return PythonBoolVector(self._bits | ~other._bits, length=self._length)

    @BackendVector.lt.define
    def lt(self, other: 'BackendVector[_BackendValue]') -> 'PythonBoolVector':
        if not isinstance(other, PythonObjectVector):
            other = self.new(bool, other)
        return PythonBoolVector(~self._bits & other._bits, length=self._length)

    @BackendVector.gt.define
    def gt(self, other: 'BackendVector[_BackendValue]') -> 'PythonBoolVector':
        if not isinstance(other, PythonObjectVector):
            other = self.new(bool, other)
        return PythonBoolVector(self._bits & ~other._bits, length=self._length)

    def and_(self, other: 'PythonBoolVector') -> 'PythonBoolVector':
        return PythonBoolVector(self._bits & other._bits, length=self._length)

    def or_(self, other: 'PythonBoolVector') -> 'PythonBoolVector':
        return PythonBoolVector(self._bits | other._bits, length=self._length)

    def xor(self, other: 'PythonBoolVector') -> 'PythonBoolVector':
        return PythonBoolVector(self._bits ^ other._bits, length=self._length)

    def not_(self) -> 'PythonBoolVector':
        return PythonBoolVector(~self._bits, length=self._length)


class PythonIntVector(IntBackendVector, PythonObjectVector[int]):
    _dtype = int

    @classmethod
    def sample_uniform(cls, value_type: type[int], length: int, *, minimum: int,
                       maximum: int) -> 'PythonIntVector':
        assert value_type is int
        assert length >= 0
        assert minimum < maximum
        return cls((random.randint(minimum, maximum) for _ in range(length)))

    def abs(self) -> 'PythonIntVector':
        return type(self)((abs(value) for value in self))

    def sign(self) -> 'PythonIntVector':
        signs: Iterable[int] = (1 if value > 0 else -1 if value < 0 else 0 for value in self._values)
        return type(self)(signs)

    def neg(self) -> 'PythonIntVector':
        return type(self)((-value for value in self._values))

    def add(self, other: 'int | PythonIntVector') -> 'PythonIntVector':
        if isinstance(other, PythonIntVector):
            return type(self)((mine + yours for mine, yours in zip(self._values, other._values)))
        else:
            return type(self)((mine + other for mine in self._values))

    def sub(self, other: 'int | PythonIntVector') -> 'PythonIntVector':
        if isinstance(other, PythonIntVector):
            return type(self)((mine - yours for mine, yours in zip(self._values, other._values)))
        else:
            return type(self)((mine - other for mine in self._values))

    def mul(self, other: 'int | PythonIntVector') -> 'PythonIntVector':
        if isinstance(other, PythonIntVector):
            return type(self)((mine * yours for mine, yours in zip(self._values, other._values)))
        else:
            return type(self)((mine * other for mine in self._values))

    @IntBackendVector.truediv.define
    def truediv(self, other: 'int | PythonIntVector') -> 'PythonFloatVector':
        if isinstance(other, PythonIntVector):
            return PythonFloatVector((mine / yours for mine, yours in zip(self._values, other._values)))
        else:
            return PythonFloatVector((mine / other for mine in self._values))

    @IntBackendVector.floordiv.define
    def floordiv(self, other: 'int | PythonIntVector') -> 'PythonIntVector':
        if isinstance(other, PythonIntVector):
            return type(self)((mine // yours for mine, yours in zip(self._values, other._values)))
        else:
            return type(self)((mine // other for mine in self._values))

    @IntBackendVector.mod.define
    def mod(self, other: 'int | PythonIntVector') -> 'PythonIntVector':
        if isinstance(other, PythonIntVector):
            return type(self)((mine % yours for mine, yours in zip(self._values, other._values)))
        else:
            return type(self)((mine % other for mine in self._values))


class PythonFloatVector(FloatBackendVector, PythonObjectVector[float]):
    _dtype = float

    @classmethod
    def sample_uniform(cls, value_type: type[float], length: int, *, minimum: float,
                       maximum: float) -> 'PythonFloatVector':
        assert value_type is float
        assert length >= 0
        assert minimum < maximum
        return cls((random.uniform(minimum, maximum) for _ in range(length)))

    @classmethod
    def sample_gaussian(cls, value_type: type[float], length: int, *, mean: float,
                        stddev: float) -> 'PythonFloatVector':
        assert value_type is float
        assert length >= 0
        return cls((random.gauss(mean, stddev) for _ in range(length)))

    def sum(self) -> float:
        return sum(self._values)

    def mean(self) -> float:
        return self.sum() / len(self._values)

    def stddev(self, mean: float = None) -> float:
        if mean is None:
            mean = self.mean()
        return (sum((value - mean) ** 2 for value in self._values) / len(self._values)) ** 0.5

    def abs(self) -> 'PythonFloatVector':
        return type(self)((abs(value) for value in self._values))

    def sign(self) -> 'PythonFloatVector':
        return type(self)((1.0 if value > 0.0 else -1.0 if value < 0.0 else 0.0 for value in self._values))

    def neg(self) -> 'PythonFloatVector':
        return type(self)((-value for value in self._values))

    def add(self, other: 'float | PythonFloatVector') -> 'PythonFloatVector':
        if isinstance(other, PythonFloatVector):
            return type(self)((mine + yours for mine, yours in zip(self._values, other._values)))
        else:
            return type(self)((mine + other for mine in self._values))

    def sub(self, other: 'float | PythonFloatVector') -> 'PythonFloatVector':
        if isinstance(other, PythonFloatVector):
            return type(self)((mine - yours for mine, yours in zip(self._values, other._values)))
        else:
            return type(self)((mine - other for mine in self._values))

    def mul(self, other: 'float | PythonFloatVector') -> 'PythonFloatVector':
        if isinstance(other, PythonFloatVector):
            return type(self)((mine * yours for mine, yours in zip(self._values, other._values)))
        else:
            return type(self)((mine * other for mine in self._values))

    @FloatBackendVector.truediv.define
    def truediv(self, other: 'float | PythonFloatVector') -> 'PythonFloatVector':
        if isinstance(other, PythonFloatVector):
            return type(self)((mine / yours for mine, yours in zip(self._values, other._values)))
        else:
            return type(self)((mine / other for mine in self._values))

    @FloatBackendVector.floordiv.define
    def floordiv(self, other: 'float | PythonFloatVector') -> 'PythonFloatVector':
        if isinstance(other, PythonFloatVector):
            return type(self)((mine // yours for mine, yours in zip(self._values, other._values)))
        else:
            return type(self)((mine // other for mine in self._values))

    @FloatBackendVector.mod.define
    def mod(self, other: 'float | PythonFloatVector') -> 'PythonFloatVector':
        if isinstance(other, PythonFloatVector):
            return type(self)((mine % yours for mine, yours in zip(self._values, other._values)))
        else:
            return type(self)((mine % other for mine in self._values))


PYTHON_BACKEND.validate()
