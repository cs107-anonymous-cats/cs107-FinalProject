import numpy as np

class DualNum:
    def __init__(self, val, der, seed = np.array([1])):
        self.val = val
        self.der = der

    # Overload addition
    def __add__(self, other): 
        try:
            return DualNum(self.val + other.val, self.der + other.der) 
        except:
            other = DualNum(other, 0)
            return DualNum(self.val + other.val, self.der + other.der)

    # Overload radd
    def __radd__(self, other):
        return self.__add__(other)

    # Overload multiplication
    def __mul__(self, other):
        try:
            return DualNum(self.val * other.val, self.val * other.der + other.val * self.der)
        except:
            other = DualNum(other, 0)
            return DualNum(self.val * other.val, self.val * other.der + other.val * self.der)

    # Overload rmul
    def __rmul__(self, other):
        return self.__mul__(other)

    # Overload subtraction
    def __sub__(self, other):
        return self.__add__(-other)

    # Overload rsub
    def __rsub__(self, other):
        return (-self).__add__(other)

    # Overload division
    def __truediv__(self, other):
        try:
            return DualNum(self.val / other.val, (self.der * other.val - self.val * other.der)/(other.val**2))
        except:
            other=DualNum(other,0)
            return DualNum(self.val / other.val, (self.der * other.val - self.val * other.der)/(other.val**2))

    # Overload rdiv
    def __rtruediv__(self, other):
         try:
             return DualNum(other.val / self.val, (other.der * self.val - other.val * self.der)/(self.val**2))
         except:
             other=DualNum(other,0)
             return DualNum(other.val / self.val, (other.der * self.val - other.val * self.der)/(self.val**2))

    # Overload negation
    def __neg__(self):
        return DualNum(-self.val, -self.der)

    # Overload power
    def __pow__(self, exponent):
        try:
            return DualNum(self.val ** exponent.val, np.exp(exponent.val * np.log(self.val)) * (exponent.der * np.log(self.val) + (exponent.val / self.val) * self.der))
        except:
            exponent=DualNum(exponent,0)
            return DualNum(self.val ** exponent.val, np.exp(exponent.val * np.log(self.val)) * (exponent.der * np.log(self.val) + (exponent.val / self.val) * self.der))

    # Overload rpow
    def __rpow__(self, exponent):
        try:
          return DualNum(exponent.val ** self.val, np.exp(self.val * np.log(exponent.val)) * (self.der * np.log(exponent.val) + (self.val / exponent.val) * exponent.der))
        except:
          exponent=DualNum(exponent,0)
          return DualNum(exponent.val ** self.val, np.exp(self.val * np.log(exponent.val)) * (self.der * np.log(exponent.val) + (self.val / exponent.val) * exponent.der))
    
    # Overload equal
    def __eq__(self, other):
	try:
	  output = (self.val == other.val) and np.array_equal(self.der, other.der)
	except AttributeError:
	# output is false because scalars are not equal to variables
	  output = False
	return output

    # Overload not equal
    def __ne__(self, other):
	return not self.__eq__(other)

    # Overload sin
    @staticmethod  
    def sin(other):
        try:
            return DualNum(np.sin(other.val), np.cos(other.val)*other.der)
        except:
            other=DualNum(other,0)
            return DualNum(np.sin(other.val), np.cos(other.val)*other.der)

    # Overload cos
    @staticmethod
    def cos(other):
        try:
            return DualNum(np.cos(other.val), -np.sin(other.val)*other.der)
        except:
            other=DualNum(other,0)
            return DualNum(np.cos(other.val), -np.sin(other.val)*other.der)

    # Overload exp
    @staticmethod
    def exp(other):
        try:
            return DualNum(np.exp(other.val), np.exp(other.val)*other.der)
        except:
            other=DualNum(other,0)
            return DualNum(np.exp(other.val), np.exp(other.val)*other.der)

    # Overload ln
    @staticmethod
    def ln(other):
        try:
            return DualNum(np.log(other.val), 1/other.val*other.der)
        except:
            other=DualNum(other,0)
            return DualNum(np.log(other.val), 1/other.val*other.der)

    # Overload tan
    @staticmethod
    def tan(other):
	try:
	    checkdomain = other.var % np.pi == (np.pi/2)
	    if checkdomain:
		raise ValueError(
		    'Cannot take tangents of multiples of pi/2 + (pi * n), where n is a positive integer')
	    new_other = np.tan(other.var)
	    tan_deriv = 1 / np.power(np.cos(other.der), 2)
	    new_der = other.der * tan_deriv
	    
            tan = DualNum(new_other, new_der)

	    return tan

        except AttributeError:
	    return np.tan(other)


"""
#Some simple tests
y=3/2**DualNum(1,1)
print(y.val,y.der)
"""



