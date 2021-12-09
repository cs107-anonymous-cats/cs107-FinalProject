import pytest
from ReverseMode import *


# tested mult,exp,add,neg,y.der,rsub,
# # adding radd didn't help
# rneg? 
def test_exp():
    x1=Node(0)
    x2=Node(1)
    y=x1*x2+Node.exp(-1*(-x1))-1
    y.reverse()
    assert y.val == 0.0 and y.getgrad(x2) == 0.0 and y.der['1'] == 1

def test_pow(): #not really adding
    x1=Node(1)
    x2=Node(2)
    y=-x1**x2
    y.reverse()
    assert y.val == -1

# tested cos,sin,pow, div, reverse, sub,ln,
# rmul(didn't work),rtruediv 
def test_vector():
    x1=Node(1)
    x2=Node(1)
    y1=4*x1*Node.cos(x2)*Node.exp(x1)-Node.sin(x1)*Node.ln(x2)
    y2=5*x1**2/x2+2+3**x1/1
    f=NodeVec([y1,y2])
    f.reverse()
    assert f.getgrad(2,x2) == -5.0

def test_variable_invalid():
    with pytest.raises(Exception):
        x1=Node(1)
        x2=1
        y1=4*x1*Node.cos(x2)*Node.exp(x1)-Node.sin(x1)*Node.ln(x2)
        y2=5*x1**2/x2+2+3**x1/1
        f=NodeVec([y1,y2])
        f.reverse()
        f.getgrad(1,x2)

def test_index_out():
    with pytest.raises(Exception):
        x1=Node(1)
        x2=1
        y1=4*x1*Node.cos(x2)*Node.exp(x1)-Node.sin(x1)*Node.ln(x2)
        y2=5*x1**2/x2+2+3**x1/1
        f=NodeVec([y1,y2])
        f.reverse()
        f.getgrad(3,x1)

def test_variable_invalid2():
    with pytest.raises(Exception):
        x1=Node(0)
        x2=Node(1)
        y=x1*x2+Node.exp(-1*(-x1))-1
        y.reverse()
        y.getgrad(3)

def test_neg_val():
    with pytest.raises(Exception):
        x1=Node(-1)
        x2=Node(2)
        y=-x1**x2

