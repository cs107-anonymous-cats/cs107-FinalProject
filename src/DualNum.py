import numpy as np
import re

class AutoDiff:
    '''
    Initiates a function variable
    
    Inputs
    ======
    self: AutoDiff object
    val: int or float; value of the user defined function(s) f evaluated at x = val
    func: function(s) to be differentiated
    seed: int, list, or array: the seed vector/derivates from parents
    Notes
    =====
    For non-default seeds, the seed for forward mode is a dictionary of ints for each variable values, and the seed for reverse mode is a list of ints for each function.
    '''
    def __init__(self, val, func, seed = None):
        forward_mode = True
        if len(val) < len(func):
            print('The number of variables < The number of functions. Use forward mode.')
        elif len(val) > len(func):
            print('The number of variables > The number of functions. Use reverse mode.')
        else:
            print('The number of variables = The number of functions. Use forward mode.')
    
        if forward_mode:
            if (seed is not None) and (not isinstance(seed, dict)):
                raise AttributeError('Forward mode needs a dictionary as a seed.')
            output = Forward(val, func, seed)
        else: 
            if (seed is not None) and (not isinstance(seed, list)):
                raise AttributeError('Reverse mode needs a list as a seed with a seed for each function input.')
            output = Reverse(val, func, seed)
            
        self.output = output
        self.val = output.val
        self.der = output.der
    
    def __repr__(self):
        return self.output.__repr__()
    
    def __str__(self):
        return self.output.__str__()   

class Forward():
    '''
    Creates a Forward AutoDiff Class
    Inputs
    ======
    self: Forward AutoDiff object
    val: int or float; value of the user defined function(s) f evaluated at x = val
    func: function(s) to be differentiated (Dual or list)
    seed: int, list, or array: the seed vector/derivates from parents
    '''
    
    def __init__(self, val, func, seed = None):
        
        # check if functions are strings
        str_check = [1 if isinstance(i, str) else 0 for i in func]
        if len(str_check) != sum(str_check):
            raise TypeError('Each function must be a string.')
        else:
            if isinstance(val, dict):
                if seed is None:
                    # create a default seed of 1 if seed is None
                    seed = {val_name: 1 for val_name in list(val.keys())}
                
                # initializing all variable values
                num_val = len(val)
                self.val2idx = dict(zip(list(val.keys()), list(range(num_val))))
                for val_name, val_value in val.items():
                    der_ = np.zeros((num_val,))
                    der_[self.val2idx[val_name]] = float(seed[val_name])
                    exec(f'{val_name} = DualNum(float(val_value), der_)')
                
                # initializing all functions
                # static methods
                static_methods = ['log', 'sqrt', 'exp', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']

                funcs = []
                for function in func:
                    for i in static_methods:
                        if i in function:
                            # for multiple but same static methods 
                            function = re.sub(i + r'\(', 'DualNum.' + i + '(', function)
                            function = re.sub('arcDualNum.', 'arc', function)
                    funcs.append(eval(function))

                self.val = np.array([i.val for i in funcs])
                self.der = np.array([i.der for i in funcs])

            else: 
                raise TypeError('The variable should be a dictionary.')

    
    def __repr__(self):
        output_str = 'Values \n'
        for val_idx in range(self.val.shape[0]):
            output_str =  output_str + 'Function F' + str(val_idx + 1) + ': ' + str(self.val[val_idx]) + '\n'

        output_str = output_str + 'Gradients \n'

        for func_idx in range(self.der.shape[0]):
            output_str = output_str + 'Function F' + str(func_idx + 1) + ': ' + str(self.der[func_idx]) + '\n'

        return output_str

    def __str__(self):
        output_str = 'Values \n'
        for val_idx in range(self.val.shape[0]):
            output_str =  output_str + 'Function F' + str(val_idx + 1) + ': ' + str(self.val[val_idx]) + '\n'

        output_str = output_str + 'Gradients \n'

        for func_idx in range(self.der.shape[0]):
            output_str = output_str + 'Function F' + str(func_idx + 1) + ': ' + str(self.der[func_idx]) + '\n'

        return output_str
    
    

class DualNum:
    '''
    Creates a Dual Class that supports Auto Differentiation custom operations
    
    Inputs
    ======
    self: DualNum object
    val: int or float; value of the user defined function(s) f evaluated at x = val
    der: int or float; value of the initialized derivative/gradient/Jacobian of user defined function(s) f
    seed: int, list, or array; the seed vector/derivative from parents
    '''


    def __init__(self, val, der, seed = np.array([1])):
        if isinstance(val, (int, float)):
            self.val = val
        else:
            raise TypeError('Did not enter valid int or float.')
        if isinstance(seed, (int, float)):
            seed = np.array([seed])
        elif isinstance(seed, list):
            seed = np.array(seed)
        self.der = seed
        
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
            if exponent.val!=None:
                assert self.val>=0, 'Cannot have (dual) number with negative value raised to a dual number power since encounter log(value) in derivative'
                return DualNum(self.val ** exponent.val, self.val**exponent.val * (exponent.der * np.log(self.val) + (exponent.val / self.val) * self.der))
        except:
            exponent=DualNum(exponent,0)
            return DualNum(self.val ** exponent.val, self.val**exponent.val * ((exponent.val / self.val) * self.der))

    # Overload rpow
    def __rpow__(self, exponent):
        try:
            return DualNum(exponent.val ** self.val, exponent.val**self.val * (self.der * np.log(exponent.val) + (self.val / exponent.val) * exponent.der))
        except:
            exponent=DualNum(exponent,0)
            return DualNum(exponent.val ** self.val, exponent.val**self.val * (self.der * np.log(exponent.val) + (self.val / exponent.val) * exponent.der))
    
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
    def exp(other, base=np.e):
        try:
            return DualNum(base**other.val, base**other.val*other.der)
        except:
            other=DualNum(other,0)
            return DualNum(base**other.val, base**other.val*other.der)

    # Overload ln
    @staticmethod
    def log(other, base=np.e):
        try:
            return DualNum(np.log(other.val)/np.log(base), 1/other.val*other.der/np.log(base))
        except:
            other=DualNum(other,0)
            return DualNum(np.log(other.val)/np.log(base), 1/other.val*other.der/np.log(base))

    # Overload tan
    @staticmethod
    def tan(other):
        try:
            checkdomain = other.val % np.pi == (np.pi/2)
            if checkdomain:
                raise ValueError('Cannot take tangents of multiples of pi/2 + (pi * n), where n is a positive integer')
            new_other = np.tan(other.val)
            tan_deriv = 1 / np.power(np.cos(other.der), 2)
            new_der = other.der * tan_deriv
        
            tan = DualNum(new_other, new_der)
        
            return tan

        except AttributeError:
            return np.tan(other)

    # Overload arcsin
    @staticmethod
    def arcsin(other):
        try:
            if other.val > 1 or other.val <-1:
                raise ValueError('please use value between -1 and 1, inclusive')
            else:
                new_other = np.arcsin(other.val)
                new_der = 1 / np.sqrt(1 - other.val**2)
        
            arcsin = DualNum(new_other, new_der)
            return arcsin
    
        except AttributeError:
            return np.arcsin(other)
    
    # Overload arccos
    @staticmethod
    def arccos(other):
        try:
            if other.val > 1 or other.val <-1:
                raise ValueError('please use value between -1 and 1, inclusive')
            else:
                new_other = np.arccos(other.val)
                new_der = -1 / np.sqrt(1 - other.val**2)
        
            arccos = DualNum(new_other, new_der)
            return arccos
    
        except AttributeError:
            return np.arccos(other)

    # Overload arctangent
    @staticmethod
    def arctan(other):
        try:
            new_other = np.arctan(other.val)
            arctan_deriv = 1 / (1 + np.power(other.val, 2))
            new_der = other.der * arctan_deriv
            
            arctan = DualNum(new_other, new_der)
            return arctan
    
        except AttributeError:
            return np.arctan(other)


    # Overload sinh
    @staticmethod
    def sinh(other):
        try:
            new_other = np.sinh(other.val)
            sinh_deriv = np.cosh(other.val)
            new_der = other.der * sinh_deriv
 
            sinh = DualNum(new_other, new_der)
            return sinh
        except AttributeError:
            return np.sinh(other) 

    # Overload cosh
    @staticmethod
    def cosh(other):
        try:
            new_other = np.cosh(other.val)
            cosh_deriv = np.sinh(other.val)
            new_der = other.der * cosh_deriv
            
            cosh = DualNum(new_other, new_der)
            return cosh
        except AttributeError:
            return np.cosh(other)

    # Overload tanh
    @staticmethod
    def tanh(other):
        try:
            new_other = np.tanh(other.val)
            tanh_deriv = 1 / np.power(np.cosh(other.val), 2)
            new_der = other.der * tanh_deriv
            
            tanh = DualNum(new_other, new_der)
            return tanh
        
        except AttributeError:
            return np.tanh(other)

    # Overload sqrt
    @staticmethod
    def sqrt(other):
        if isinstance(other, DualNum):
            new_other = np.sqrt(other.val)
            new_der = 0.5 / np.sqrt(other.val) * other.der
            return DualNum(new_other, new_der)
        elif other < 0:
            raise ValueError('Cannot take square roots of negative values')
        else:
            return np.sqrt(other)
    
    
# simple tests
print('Example with AutoDiff')
val = {'x': 0, 'y': 0.25}
func = ['cos(x) + y ** 2', '2 * sin(x) - sqrt(y)/3', '3 * sinh(y) - 4 * arcsin(y) + 5', 'cosh(x) / sinh(y)']

output = AutoDiff(val, func)
print(output)

print('Example with Forward')
val = {'x': 1, 'y': 2}
func = ['cos(x) + y ** 2', 'tan(x)/3 - sqrt(y)']
output = Forward(val, func)
print(output)

print('Example with Seed')
val = {'x': 10, 'y': 3}
seed = {'x': 1, 'y': 2}
func = ['cos(x) + y ** 2', 'tan(x)/3 - sqrt(y)']
output = AutoDiff(val, func, seed)
print(output)

