import pytest
import torch
from cmrxrecon.dl.lowrank_varnet import cg_data_consistency

# Dummy system matrix functions for tests
def dummy_system_matrix(p, sensetivity, basis, mask):
    return p * 2

def identity_system_matrix(p, sensetivity, basis, mask):
    return p

def diagonal_system_matrix(p, sensetivity, basis, mask):
    return torch.matmul(torch.tensor([[2.0, 0.0], [0.0, 3.0]]), p)

def complex_valued_system_matrix(p, sensetivity, basis, mask):
    matrix = torch.tensor([[2.0 + 1.0j, 0.0], [0.0, 3.0 + 2.0j]], dtype=torch.complex64)
    return torch.matmul(matrix, p)

@pytest.mark.parametrize("b,expected_result", [
    (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.5, 1.0, 1.5])),
    (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])),
])
def test_solve_dummy_system_matrix(b, expected_result):
    cg = cg_data_consistency(iterations=10)
    result = cg.solve(dummy_system_matrix, b, None, None, None)
    assert torch.allclose(result, expected_result, atol=1e-5)

@pytest.mark.parametrize("b,expected_result", [
    (torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])),
    (torch.tensor([-1.0, 1.0]), torch.tensor([-1.0, 1.0])),
])
def test_solve_identity_system_matrix(b, expected_result):
    cg = cg_data_consistency(iterations=10)
    result = cg.solve(identity_system_matrix, b, None, None, None)
    assert torch.allclose(result, expected_result, atol=1e-5)

@pytest.mark.parametrize("b,expected_result", [
    (torch.tensor([1.0, 1.0]), torch.tensor([1/2.0, 1/3.0])),
    (torch.tensor([4.0, 6.0]), torch.tensor([2.0, 2.0])),
])
def test_solve_complex_system_matrix(b, expected_result):
    cg = cg_data_consistency(iterations=10)
    result = cg.solve(diagonal_system_matrix, b, None, None, None)
    assert torch.allclose(result, expected_result, atol=1e-5)

@pytest.mark.parametrize("b,expected_result", [
    (torch.tensor([0.6 + 0.8j, 1.3 + 0.0j], dtype=torch.complex64), torch.tensor([0.4 + 0.2j, 0.3 - 0.2j], dtype=torch.complex64)),
])
def test_complex_valued_matrix(b, expected_result):
    cg = cg_data_consistency(iterations=10)
    result = cg.solve(complex_valued_system_matrix, b, None, None, None)
    assert torch.allclose(result, expected_result, atol=1e-5)


def test_solve_convergence():
    cg = cg_data_consistency(iterations=10, error_tolerance=1e-8)
    b = torch.tensor([1.0, 2.0, 3.0])
    result = cg.solve(dummy_system_matrix, b, None, None, None)
    assert torch.allclose(result, b / 2.0, atol=1e-8)


if __name__ == "__main__":
    pytest.main()

