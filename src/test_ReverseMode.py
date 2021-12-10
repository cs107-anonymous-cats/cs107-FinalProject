import pytest
from ReverseMode import *

# tested mult,exp,add,neg,y.der,rsub,
# tested cos,sin,pow, div, reverse, sub,log,rtruediv
# tested cosh,sinh,tanh,arccos,arcsin,arctan,sqrt,logistic
# adding radd/rmul/rpow/pow didn't help

def test_exp():
    x1=Node(0)
    x2=Node(1)
    y=x1*x2+Node.exp(-1*(-x1))-1
    y.reverse()
    assert y.val == 0.0 and y.getgrad(x2) == 0.0 and y.der['1'] == 1


def test_radd_rmul_rsub_rpow():
    x2=Node(2)
    y1=2+x2
    y2=2*x2
    y3=2-x2
    y4=2**x2
    assert y1.val == 4 and y2.val == 4 and y3.val == 0 and y4.val == 4

def test_pow(): 
    x1=Node(1)
    x2=Node(2)
    y=-x1**x2
    y.reverse()
    assert y.val == -1

def test_logistic():
    x1=Node(0)
    y1=4*Node.logistic(x1)
    y1.reverse()
    assert y1.val == 2.0

def test_vector():
    x1=Node(1)
    x2=Node(1)
    y1=4*x1*Node.cos(x2)*Node.exp(x1)-Node.sin(x1)*Node.log(x2)
    y2=5*x1**2/x2+2+3**x1/1
    f=NodeVec([y1,y2])
    f.reverse()
    assert f.getgrad(2,x2) == -5.0

def test_vector2():
    x1=Node(1)
    x2=Node(1)
    y1=4*x1*Node.arccos(x2)*Node.cosh(x1)-Node.arcsin(x1)*Node.arctan(x2)+Node.sinh(x1)*Node.tanh(x2)
    y2=5*x1**2/x2+2+3**x1/1
    f=NodeVec([y1,y2])
    f.reverse()
    assert f.getgrad(2,x2) == -5.0

def test_sqrt():
    x1=Node(1)
    y1=4*Node.sqrt(x1)
    y1.reverse()
    assert y1.val == 4.0

def test_variable_invalid():
    with pytest.raises(Exception):
        x1=Node(1)
        x2=1
        y1=4*x1*Node.cos(x2)*Node.exp(x1)-Node.sin(x1)*Node.log(x2)
        y2=5*x1**2/x2+2+3**x1/1
        f=NodeVec([y1,y2])
        f.reverse()
        f.getgrad(1,x2)

def test_index_out():
    with pytest.raises(Exception):
        x1=Node(1)
        x2=1
        y1=4*x1*Node.cos(x2)*Node.exp(x1)-Node.sin(x1)*Node.log(x2)
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

def test_sqrt2():
    with pytest.raises(Exception):
        x1=Node(-1)
        y1=4*Node.sqrt(x1)



