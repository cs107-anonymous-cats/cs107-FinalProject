from AutoDiff import *
import pytest


def test_autodiff3():
    seed=[1,0]
    x1=DualNum(1,seed[0])
    x2=DualNum(2,seed[1])
    x3 = DualNum(1,1)
    y=DualNumVec([x1*x2+DualNum.exp(x1/x2),x1**2+x2**2])
    y2=DualNumVec([x1/2,2/x2])
    y3=DualNum.tan(x1) + DualNum.exp(2) + DualNum.arcsin(x3) + DualNum.arccos(x3)+ DualNum.arctan(x3)
    y4=DualNum.sinh(x1)+DualNum.cosh(x1)+ DualNum.tanh(x1)+DualNum.tanh(1)+DualNum.sqrt(x1)
    print(y.getvals(),y.getgrad())

def test_neg_sqrt():
    with pytest.raises(ValueError):
        DualNum.sqrt(-1)

def test_neg_arccos():
    with pytest.raises(ValueError):
        DualNum.arccos(DualNum(-3,1))
        DualNum.arcsin(DualNum(-3,1))

def test_neg_arcsin():
    with pytest.raises(ValueError):
        DualNum.arcsin(DualNum(-3,1))

def test_tan():
    with pytest.raises(ValueError):
        DualNum.tan(DualNum(np.pi/2 + (np.pi),1))

def test_eq():
    y1=2*DualNum.sin(DualNum(0,1))+3
    y2=2*DualNum.sin(DualNum(0,1))+3
    y3 = 10/DualNum(1,1)
    print(y1 == y2, y1 != y3, y1 == 1)

def test_ne():
    y1=2*DualNum.sin(DualNum(0,1))+3
    y3 = 10/DualNum(1,1)
    print(y1 != y3)

def test_sin_cos_arctan():
    y1=2*DualNum.sin(0)+3
    y2=2*DualNum.cos(0)+3
    y3=2*DualNum.arctan(0)+3
    y4=2*DualNum.sinh(0)+3
    y4=2*DualNum.cosh(0)+3
    
def test_rmul():
    y=DualNum.sin(DualNum(0,1))*2+3
    assert y.val == 3.0 

def test_exp():
    y=2*DualNum.exp(DualNum(0,1))+3
    assert y.val == 5.0 
    y=2*DualNum.exp(0)+3
    assert y.val == 5.0 

def test_log():
    y=2*DualNum.log(DualNum(1,1))+3
    assert y.val == 3.0
    y=2*DualNum.log(1)+3
    assert y.val == 3.0

def test_pow2():
    y=3**DualNum(1,0)+5**DualNum(1,0)
    assert y.val == 8

def test_radd():
    y = 10 + DualNum(1,1)
    assert y.val == 11 

def test_neg():
    y = DualNum.__neg__(DualNum(1,1))
    assert y.val == -1

def test_rtruediv():
    y = 10/DualNum(1,1)
    assert y.val == 10.0


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
    y1=4*x1*Node.cos(x2)*Node.exp(x1)-Node.sin(x1)*Node.log(x2)+Node.arctan(1)
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

def test_sqrt2():
    with pytest.raises(Exception):
        x1=Node(-1)
        y1=4*Node.sqrt(x1)

