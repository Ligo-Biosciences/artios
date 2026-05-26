import torch

from artios.utils.monitoring import grad_norm


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.bias = torch.nn.Parameter(torch.zeros(2))
        self.no_grad_param = torch.nn.Parameter(torch.zeros(1))


def test_grad_norm_matches_flattened_parameter_gradients_without_scalar_rebuild(
    monkeypatch,
):
    module = TinyModule()
    module.weight.grad = torch.tensor([[3.0, 4.0], [0.0, 12.0]])
    module.bias.grad = torch.tensor([5.0, 0.0])
    module.no_grad_param.grad = None
    make_tensor = torch.tensor

    def fail_if_scalar_list_is_rebuilt(*args, **kwargs):
        raise AssertionError("grad_norm should not rebuild CUDA scalar lists")

    monkeypatch.setattr(torch, "tensor", fail_if_scalar_list_is_rebuilt)

    norms, total_norm = grad_norm(module, norm_type=2, group_separator=".")

    assert torch.equal(norms["grad_2.0_norm.weight"], make_tensor(13.0))
    assert torch.equal(norms["grad_2.0_norm.bias"], make_tensor(5.0))
    assert "grad_2.0_norm.no_grad_param" not in norms
    assert total_norm is norms["grad_2.0_norm_total"]
    assert torch.allclose(total_norm, torch.sqrt(make_tensor(194.0)))


def test_grad_norm_supports_infinity_norm():
    module = TinyModule()
    module.weight.grad = torch.tensor([[1.0, -7.0], [3.0, 4.0]])
    module.bias.grad = torch.tensor([-5.0, 2.0])

    norms, total_norm = grad_norm(module, norm_type=float("inf"))

    assert torch.equal(norms["grad_inf_norm/weight"], torch.tensor(7.0))
    assert torch.equal(norms["grad_inf_norm/bias"], torch.tensor(5.0))
    assert torch.equal(total_norm, torch.tensor(7.0))
