from DualNum import *
import pytest

# tested: cos, sin, sqrt, sinh, arcsin, cosh, sinh, tan
# tested: mul, add, radd, sub, div, rdiv, pow, rpow, exp, 
# tested: eq, ne, arcos, arctan, tanh, sqrt, 

def test_autodiff():
    val = {'x': 0, 'y': 0.25}
    func = ['cos(x) + y ** 2', '2 * sin(x) - sqrt(y)/3', '3 * sinh(y) - 4 * arcsin(y) + 5', 'cosh(x) / sinh(y)','tan(x)/3 - sqrt(y)']
    output = AutoDiff(val, func)
    print(output)

def test_autodiff2():
    val = {'x': 0, 'y': 0.25}
    func = ['arccos(x) + y ** 2', '2 * arctan(x) - tanh(y)/3', '3 * sinh(y) - 4 * arcsin(y) + 5', 'cosh(x) / sinh(y)','tan(x)/3 - sqrt(y)']
    output = AutoDiff(val, func)
    repr(output)

def test_with_seed(): 
    val = {'x': 10, 'y': 3}
    seed = {'x': 1, 'y': 2}
    func = ['cos(x) + y ** 2', 'tan(x)/3 - sqrt(y)']
    output = AutoDiff(val, func, seed)

def test_functionstring():
    with pytest.raises(TypeError):
        val = {'x': 1, 'y': 2}
        func = ['cos(x) + y ** 2', 'tan(x)/3 - sqrt(y)',2]
        output = Forward(val, func)

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

def test_pow(): 
    with pytest.raises(TypeError):
        DualNum(-1,1)**DualNum(1,1)

def test_init(): 
    with pytest.raises(TypeError):
        DualNum([1],1)


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





