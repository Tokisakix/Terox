import pytest
import math

from terox.autodiff.function import add, sub, mul, div, inv, max, min, eq, lt, gt, abs, exp, log, relu

@pytest.mark.test_function
def test_add(a:float=2.0, b:float=3.0) -> None:
    res = add(a, b)
    ref = a + b
    assert res == ref
    return

@pytest.mark.test_function
def test_sub(a:float=2.0, b:float=3.0) -> None:
    res = sub(a, b)
    ref = a - b
    assert res == ref
    return

@pytest.mark.test_function
def test_mul(a:float=2.0, b:float=3.0) -> None:
    res = mul(a, b)
    ref = a * b
    assert res == ref
    return

@pytest.mark.test_function
def test_div(a:float=2.0, b:float=3.0) -> None:
    res = div(a, b)
    ref = a / b
    assert res == ref
    return

@pytest.mark.test_function
def test_inv(a:float=2.0) -> None:
    res = inv(a)
    ref = 1.0 / a
    assert res == ref
    return

@pytest.mark.test_function
def test_max(a:float=2.0, b:float=3.0) -> None:
    res = max(a, b)
    ref = a if a > b else b
    assert res == ref
    return

@pytest.mark.test_function
def test_min(a:float=2.0, b:float=3.0) -> None:
    res = min(a, b)
    ref = a if a < b else b
    assert res == ref
    return

@pytest.mark.test_function
def test_eq(a:float=2.0, b:float=3.0) -> None:
    res = eq(a, b)
    ref = 1.0 if a == b else 0.0
    assert res == ref
    return

@pytest.mark.test_function
def test_lt(a:float=2.0, b:float=3.0) -> None:
    res = lt(a, b)
    ref = 1.0 if a < b else 0.0
    assert res == ref
    return

@pytest.mark.test_function
def test_gt(a:float=2.0, b:float=3.0) -> None:
    res = gt(a, b)
    ref = 1.0 if a > b else 0.0
    assert res == ref
    return

@pytest.mark.test_function
def test_abs(a:float=-2.0) -> None:
    res = abs(a)
    ref = a if a > 0.0 else -a
    assert res == ref
    return

@pytest.mark.test_function
def test_exp(a:float=2.0) -> None:
    res = exp(a)
    ref = math.exp(a)
    assert res == ref
    return

@pytest.mark.test_function
def test_log(a:float=2.0) -> None:
    res = log(a)
    ref = math.log(a)
    assert res == ref
    return

@pytest.mark.test_function
def test_relu(a:float=-2.0) -> None:
    res = relu(a)
    ref = a if a > 0.0 else 0.0
    assert res == ref
    return