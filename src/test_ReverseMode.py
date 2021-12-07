import pytest

def test_exp():
    x1=Node(1)
    x2=Node(2)
    y=x1*x2+Node.exp(x1*x2)