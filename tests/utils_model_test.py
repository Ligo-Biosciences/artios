import pytest
import torch
import numpy as np
from artios.model.utils import add, mul


class TestArithmeticOperations:
    @pytest.fixture
    def tensor_data(self):
        return {
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
            'expected_sum': torch.tensor([5, 7, 9]),
            'expected_product': torch.tensor([4, 10, 18])
        }

    @pytest.fixture
    def numpy_data(self):
        return {
            'a': np.array([1, 2, 3]),
            'b': np.array([4, 5, 6]),
            'expected_sum': np.array([5, 7, 9]),
            'expected_product': np.array([4, 10, 18])
        }

    def test_add_tensors(self, tensor_data):
        # Test without in_place
        result = add(tensor_data['a'], tensor_data['b'], inplace_safe=False)
        assert torch.equal(result, tensor_data['expected_sum'])
        assert torch.equal(tensor_data['a'], torch.tensor([1, 2, 3]))
        
        # Test with in_place
        a = torch.tensor([1, 2, 3])
        result = add(a, tensor_data['b'], inplace_safe=True)
        assert torch.equal(result, tensor_data['expected_sum'])
        assert torch.equal(a, tensor_data['expected_sum'])

    def test_add_numpy(self, numpy_data):
        # Test without in_place
        result = add(numpy_data['a'], numpy_data['b'], inplace_safe=False)
        np.testing.assert_array_equal(result, numpy_data['expected_sum'])
        np.testing.assert_array_equal(numpy_data['a'], np.array([1, 2, 3]))
        
        # Test with in_place
        a = np.array([1, 2, 3])
        result = add(a, numpy_data['b'], inplace_safe=True)
        np.testing.assert_array_equal(result, numpy_data['expected_sum'])
        np.testing.assert_array_equal(a, numpy_data['expected_sum'])

    def test_mul_tensors(self, tensor_data):
        # Test without in_place
        result = mul(tensor_data['a'], tensor_data['b'], inplace_safe=False)
        assert torch.equal(result, tensor_data['expected_product'])
        assert torch.equal(tensor_data['a'], torch.tensor([1, 2, 3]))
        
        # Test with in_place
        a = torch.tensor([1, 2, 3])
        result = mul(a, tensor_data['b'], inplace_safe=True)
        assert torch.equal(result, tensor_data['expected_product'])
        assert torch.equal(a, tensor_data['expected_product'])

    def test_mul_numpy(self, numpy_data):
        # Test without in_place
        result = mul(numpy_data['a'], numpy_data['b'], inplace_safe=False)
        np.testing.assert_array_equal(result, numpy_data['expected_product'])
        np.testing.assert_array_equal(numpy_data['a'], np.array([1, 2, 3]))
        
        # Test with in_place
        a = np.array([1, 2, 3])
        result = mul(a, numpy_data['b'], inplace_safe=True)
        np.testing.assert_array_equal(result, numpy_data['expected_product'])
        np.testing.assert_array_equal(a, numpy_data['expected_product'])


if __name__ == "__main__":
    pytest.main([__file__])
