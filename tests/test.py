from src.dualnum import *
import pytest

def test_cos():
    y=2*DualNum.cos(DualNum(0,1))+3
    assert y.val == 5.0 and y.der == 0.0
