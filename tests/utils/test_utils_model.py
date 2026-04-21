import pytest
import torch

from artios.model.utils import masked_mean


class TestMaskedMean:
    def test_masked_mean_1d_axis(self):
        input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        result = masked_mean(input_tensor, mask, axes=1)
        expected = torch.tensor([2.0, 4.5])  # (1+3)/2, (4+5)/2
        torch.testing.assert_close(result, expected)

    def test_masked_mean_2d_axes(self):
        input_tensor = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        )
        mask = torch.tensor([[[1.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]])

        result = masked_mean(input_tensor, mask, axes=[1, 2])
        expected = torch.tensor([2.0, 6.5])  # (1+2+3)/3, (5+8)/2
        torch.testing.assert_close(result, expected)

    def test_masked_mean_keepdim(self):
        input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

        result = masked_mean(input_tensor, mask, axes=1, keepdim=True)
        expected = torch.tensor([[1.5], [5.5]])  # (1+2)/2, (5+6)/2
        torch.testing.assert_close(result, expected)
        assert result.shape == (2, 1)

    def test_masked_mean_broadcast_mask(self):
        input_tensor = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        )
        mask = torch.tensor([[[1.0]], [[1.0]]])  # Shape (2, 1, 1)

        result = masked_mean(input_tensor, mask, axes=[1, 2])
        expected = torch.tensor([3.5, 9.5])  # Mean of all elements in each batch
        torch.testing.assert_close(result, expected)

    def test_masked_mean_partial_broadcast(self):
        input_tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        mask = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]]]
        )  # Full shape on last dim
        print("shape of input_tensor", input_tensor.shape)
        print("shape of mask", mask.shape)

        result = masked_mean(input_tensor, mask, axes=1)
        expected = torch.tensor([[3.0, 3.0]])  # [(1+3+5)/3, (2+4)/2]
        torch.testing.assert_close(result, expected)

    def test_masked_mean_all_zeros_mask(self):
        input_tensor = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[0.0, 0.0, 0.0]])

        result = masked_mean(input_tensor, mask, axes=1)
        # With eps=1e-6, should handle division by near-zero gracefully
        assert torch.isfinite(result).all()

    def test_masked_mean_single_axis_as_int(self):
        input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

        # Test with axis as int (not list)
        result = masked_mean(input_tensor, mask, 1)  # axis as int
        expected = torch.tensor([1.5, 5.5])
        torch.testing.assert_close(result, expected)

    def test_masked_mean_mismatched_dims_raises(self):
        input_tensor = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([1.0, 0.0, 1.0])  # 1D instead of 2D

        with pytest.raises(AssertionError):
            masked_mean(input_tensor, mask, axes=0)

    def test_masked_mean_mismatched_size_raises(self):
        input_tensor = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[1.0, 0.0]])  # Wrong size on axis 1

        with pytest.raises(ValueError, match="Input and mask must have the same size"):
            masked_mean(input_tensor, mask, axes=1)

    def test_masked_mean_non_contiguous_axes(self):
        # Test with axes [0, 2] skipping axis 1
        input_tensor = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        )
        mask = torch.tensor(
            [[[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]]
        )
        print("shape of input_tensor", input_tensor.shape)
        print("shape of mask", mask.shape)

        result = masked_mean(input_tensor, mask, axes=[0, 2])

        expected = torch.tensor([4.75, 8.25])
        print("result", result)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_masked_mean_3d_non_contiguous(self):
        # 4D tensor, reduce axes [0, 3]
        input_tensor = torch.randn(2, 3, 4, 5)
        mask = torch.ones(2, 3, 4, 5)
        mask[0, :, :, 2:] = 0  # Mask out some elements
        mask[1, 1, :, :3] = 0

        result = masked_mean(input_tensor, mask, axes=[0, 3])
        assert result.shape == (3, 4)

        # Verify a specific element calculation
        manual_sum = (input_tensor[0, 0, 0, :2] * mask[0, 0, 0, :2]).sum() + (
            input_tensor[1, 0, 0, :] * mask[1, 0, 0, :]
        ).sum()
        manual_count = mask[0, 0, 0, :2].sum() + mask[1, 0, 0, :].sum()
        manual_mean = manual_sum / (manual_count + 1e-6)
        torch.testing.assert_close(result[0, 0], manual_mean, rtol=1e-5, atol=1e-5)

    def test_masked_mean_broadcast_multiple_dims(self):
        # Mask broadcasts over multiple dimensions
        input_tensor = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]
        )
        mask = torch.tensor(
            [[[[1.0]]]]
        )  # Shape (1, 1, 1, 1), broadcasts to (1, 2, 2, 2)

        result = masked_mean(input_tensor, mask, axes=[1, 2, 3])
        expected = torch.tensor([4.5])  # Mean of all 8 elements
        torch.testing.assert_close(result, expected)

        # Now with partial broadcast
        mask2 = torch.tensor([[[[1.0, 0.0]]]])  # Shape (1, 1, 1, 2)
        result2 = masked_mean(input_tensor, mask2, axes=[1, 2, 3])
        expected2 = torch.tensor([4.0])  # Mean of [1, 3, 5, 7]
        torch.testing.assert_close(result2, expected2)

    def test_masked_mean_mixed_broadcast_axes(self):
        # Complex case: different broadcast patterns on different axes
        input_tensor = torch.randn(4, 5, 6, 7)
        mask = torch.ones(4, 1, 6, 1)  # Broadcasts on axes 1 and 3

        # Reduce axes [0, 2] (non-broadcast)
        result = masked_mean(input_tensor, mask, axes=[0, 2])
        assert result.shape == (5, 7)

        # Reduce axes [1, 3] (broadcast axes)
        result2 = masked_mean(input_tensor, mask, axes=[1, 3])
        assert result2.shape == (4, 6)

        # The broadcast_factor should be 5*7 for axes [1,3]
        # Manually verify one element
        manual_sum = (input_tensor[0, :, 0, :] * mask[0, :, 0, :]).sum()
        manual_count = mask[0, :, 0, :].sum() * 35  # broadcast_factor = 5*7
        manual_mean = manual_sum / (manual_count + 1e-6)
        torch.testing.assert_close(result2[0, 0], manual_mean, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
