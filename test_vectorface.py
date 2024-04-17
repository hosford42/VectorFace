import unittest

from vectorface import get_backend, VectorBackend, activate_backend, Vector


class TestPythonBackend(unittest.TestCase):

    def test_get_backend(self):
        python_backend = get_backend('python')
        assert isinstance(python_backend, VectorBackend)

    def test_bool_vector(self):
        activate_backend('python')
        values = [False, True, True, False, True]
        vector = Vector.new(bool, values)
        assert vector.value_type is bool
        for value in vector:
            assert isinstance(value, bool)
        assert len(vector) == len(values)
        for init_value, vector_value in zip(values, vector):
            assert init_value == vector_value
        assert str(vector) == '01101'
        assert str(vector[2:]) == '101'
        assert str(vector[:-2]) == '011'
        assert str(vector[1:-1]) == '110'
        assert str(vector[-3:]) == '101'
        assert str(vector[:3]) == '011'
        assert str(reversed(vector)) == '10110'
        all_true = Vector.new(bool, [True] * len(vector))
        all_false = Vector.new(bool, [False] * len(vector))
        assert all_true == ~all_false
        assert all_true.ne(all_false).all()
        assert vector | vector == vector
        assert vector | ~vector == all_true
        assert vector & vector == vector
        assert vector & ~vector == all_false
        assert vector ^ vector == all_false
        assert vector ^ ~vector == all_true
        assert vector ^ all_false == vector
        assert vector ^ all_true == ~vector
