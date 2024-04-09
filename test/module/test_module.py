import pytest

from terox.autodiff.scalar import Scalar
from terox.module.module import Module
from terox.module.parameter import Parameter

class M1(Module):
    def __init__(self, p1:Parameter, p2:Parameter) -> None:
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        return
    
class M2(Module):
    def __init__(self, m1:M1, p3:Parameter) -> None:
        super().__init__()
        self.m1 = m1
        self.p3 = p3
        return

p1 = Parameter(Scalar(1.0))
p2 = Parameter(Scalar(2.0))
p3 = Parameter(Scalar(3.0))
m1 = M1(p1, p2)
m2 = M2(m1, p3)

@pytest.mark.test_module
def test_module_dict() -> None:
    assert m1.getParmeterDict() == {"p1":p1, "p2":p2}
    assert m2.getParmeterDict() == {"m1.p1":p1, "m1.p2":p2, "p3":p3}
    return

@pytest.mark.test_module
def test_module_parmeter() -> None:
    assert m1.parmeters() == [p1, p2]
    assert m2.parmeters() == [p1, p2, p3]
    return