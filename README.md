# Vectorface

*Vector computing with arbitrary backends*

## Links
* [Source](https://github.com/hosford42/vectorface)
* [Distribution](https://pypi.python.org/pypi/vectorface)

## License
The package is available for download under the permissive [Revised BSD License](https://github.com/hosford42/vectorface/blob/master/LICENSE).

## Description
Vectorface is a collection of standardized interfaces and adapters for vector 
computing libraries such as numpy, tensorflow, pytorch, and cupy. It is 
intended to make swapping out vector computing libraries completely transparent
to client code, and also to facilitate interoperability between them.

**Currently, only 1D vectors are supported.** This is a major limitation on the
utility of this package. Support for multidimensional arrays (matrices and 
tensors) is a top priority and will be added in a future release.

# Usage

```python
from vectorface import list_available_backends, activate_backend, Vector

print("Available backends:")
for backend in list_available_backends():
    print(f"\t{backend}")
    
activate_backend('numpy')

i = Vector.new(bool, [True, False, False, True])
j = Vector.new(int, [1, 2, 3, 4])
k = Vector.new(float, [1.1, -17.0, 3.14159])

print(f'{i=}')
print(f'{j=}')
print(f'{k=}')
print(f'{~i=}')
print(f'{i.cast(int)*j=}')
print(f'{k.exp()}')
```
