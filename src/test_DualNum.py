from DualNum import *
import pytest



def test_sin_and_mul():
    y=2*DualNum.sin(DualNum(0,1))+3
    assert y.val == 3.0 and y.der == 2.0

def test_sin2():
    y=DualNum.sin(0)*2+3
    assert y.val == 3.0 and y.der == 0.0

def test_mul2():
    y=DualNum(1,1)*DualNum(2,1)
    assert y.val == 2 and y.der == 3 

def test_rmul():
    y=DualNum.sin(DualNum(0,1))*2+3
    assert y.val == 3.0 and y.der == 2.0

def test_cos():
    y=2*DualNum.cos(DualNum(0,1))+3
    assert y.val == 5.0 and y.der == 0.0

def test_exp():
    y=2*DualNum.exp(DualNum(0,1))+3
    assert y.val == 5.0 and y.der == 2.0
    y=2*DualNum.exp(0)+3
    assert y.val == 5.0 and y.der == 0.0

def test_ln():
    y=2*DualNum.ln(DualNum(1,1))+3
    assert y.val == 3.0 and y.der == 2.0
    y=2*DualNum.ln(1)+3
    assert y.val == 3.0 and y.der == 0.0

def test_add_and_pow():
    y=10*DualNum(1,1)**3+3*DualNum(1,1)**5
    assert y.val == 13 and y.der == 45.0

def test_pow2():
    y=3**DualNum(1,0)+5**DualNum(1,0)
    assert y.val == 8 and y.der == 0.0

def test_radd():
    y = 10 + DualNum(1,1)
    assert y.val == 11 and y.der == 1

def test_sub():
    y =10-DualNum(1,1)
    assert y.val == 9 and y.der == -1

def test_rsub():
    y =DualNum(1,1)-10
    assert y.val == -9 and y.der == 1

def test_neg():
    y = DualNum.__neg__(DualNum(1,1))
    assert y.val == -1 and y.der == -1

def test_truediv():
    y = DualNum(1,1)/10
    assert y.val == 0.1 and y.der == 0.1

def test_rtruediv():
    y = 10/DualNum(1,1)
    assert y.val == 10.0 and y.der == -10.0





