import numpy as np

class DualNum:
    '''
    Creates a Dual Class that supports Auto Differentiation custom operations
    In order to generate a derivative of a function with respect to a specific variable, we initalize the variable and function(s) as DualNum objects
    
    
    Inputs
    ======
    self: DualNum object
    val: int or float; value of the user defined function(s) f evaluated at x = val
    der: int or float; value of the initialized derivative/gradient/Jacobian of user defined function(s) f
        der is an np.array for scalar functions and a list for vector functions
    seed: int, list, or array; the seed vector/derivative from parents
    
    Methods
    ======
    __add__, __radd__, __mul__, __rmul__, __sub__, __rsub__, __neg__, __pow__, __rpow__, __truediv__, __rtruediv__, 
    __eq__, __ne__, __sqrt__, __sin__, __cos__, __tan__, __arcsin__, __arccos__, __arctan__, __sinh__, __cosh__, 
    __tanh__, __logistic__, __log__, __exp__
    
    Dunder methods are overloaded to operate on DualNum objects
    
    Examples
    ========
    Scalar Input
    >>> x1=DualNum(0, 1)
    >>> x2=DualNum(1, 1)
    >>> y=x1+x2
    >>> print(y.val, y.der)
    1 2
    Scalar Input (DualNum + Scalar)
    >>> y2=x1 + 3
    >>> print(y2.val, y2.der)
    3 1
    
    Applying a function to the object
    >>> x3=2*DualNum.cos(DualNum(0,1))
    >>> print(x3.val, x3.der)
    20.0 0.0

    
    '''
    def __init__(self, val, der, seed = np.array([1])):
        '''
        Constructs attributes of a DualNum object to represent a variable or function
        
        Inputs
        ======
        self: DualNum object
        val: int or float; value of the user defined function(s) f evaluated at x = val
        der: int or float; value of the initialized derivative/gradient/Jacobian of user defined function(s) f
            der is an np.array for scalar functions and a list for vector functions
        seed: int, list, or array; the seed vector/derivative from parents

        '''
        self.val = val
        self.der = der

    # Overload addition
    def __add__(self, other): 
        """
        add two DualNum objects together

        Attributes
        ----------------------------------
        other: int, float, np.array, or DualNum
          DualNum object to add self to

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
        try:
            return DualNum(self.val + other.val, self.der + other.der) 
        except:
            other = DualNum(other, 0)
            return DualNum(self.val + other.val, self.der + other.der)

    # Overload radd
    def __radd__(self, other):
        """
        add two DualNum objects together

        Attributes
        ----------------------------------
        other: int, float, np.array, or DualNum
          DualNum object to add self to

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
        return self.__add__(other)

    # Overload multiplication
    def __mul__(self, other):
        """
        multiply two DualNum objects together

        Attributes
        ----------------------------------
        other: int, float, np.array, or DualNum
          DualNum object to multiply self to

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
        try:
            return DualNum(self.val * other.val, self.val * other.der + other.val * self.der)
        except:
            other = DualNum(other, 0)
            return DualNum(self.val * other.val, self.val * other.der + other.val * self.der)

    # Overload rmul
    def __rmul__(self, other):
        """
        perform reverse multiplication of two DualNum objects together

        Attributes
        ----------------------------------
        other: DualNum
          DualNum object to multiply other to

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
        return self.__mul__(other)

    # Overload subtraction
    def __sub__(self, other):
        """
        subtract two DualNum objects

        Attributes
        ----------------------------------
        other: int, float, np.array, or DualNum
          DualNum object to subtract other from

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """        
        return self.__add__(-other)

    # Overload rsub
    def __rsub__(self, other):
        """
        perform reverse subtraction of two DualNum objects

        Attributes
        ----------------------------------
        other: int, float, np.array, or DualNum
          DualNum object to subtract self from

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
        return (-self).__add__(other)

    # Overload division
    def __truediv__(self, other):
        """
        divide self with other

        Attributes
        ----------------------------------
        other: int, float, np.array, or DualNum
          DualNum object to divide self with

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
        try:
            return DualNum(self.val / other.val, (self.der * other.val - self.val * other.der)/(other.val**2))
        except:
            other=DualNum(other,0)
            return DualNum(self.val / other.val, (self.der * other.val - self.val * other.der)/(other.val**2))

    # Overload rdiv
    def __rtruediv__(self, other):
        """
        divide other with self

        Attributes
        ----------------------------------
        other: int, float, np.array, or DualNum
          DualNum object to divide other with

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
        try:
            return DualNum(other.val / self.val, (other.der * self.val - other.val * self.der)/(self.val**2))
        except:
            other=DualNum(other,0)
            return DualNum(other.val / self.val, (other.der * self.val - other.val * self.der)/(self.val**2))

    # Overload negation
    def __neg__(self):
        """
        negation of DualNum value

        Returns
        ---------------------------------
          DualNum object with updated value
        """
        return DualNum(-self.val, -self.der)

    # Overload power
    def __pow__(self, exponent):
        
        """
        Raise a DualNum object to the power of exponent

        Attributes
        ----------------------------------
        exponent: int, float, or DualNum
          DualNum object to be exponentiated with

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
        try:
            return DualNum(self.val ** exponent.val, np.exp(exponent.val * np.log(self.val)) * (exponent.der * np.log(self.val) + (exponent.val / self.val) * self.der))
        except:
            exponent=DualNum(exponent,0)
            return DualNum(self.val ** exponent.val, np.exp(exponent.val * np.log(self.val)) * (exponent.der * np.log(self.val) + (exponent.val / self.val) * self.der))


    # Overload rpow
    def __rpow__(self, exponent):
        """
        Raise a number to the power of DualNum object

        Attributes
        ----------------------------------
        exponent: int or float
          Exponent to which self is raised, self is a DualNum object

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
        try:
            return DualNum(exponent.val ** self.val, np.exp(self.val * np.log(exponent.val)) * (self.der * np.log(exponent.val) + (self.val / exponent.val) * exponent.der))
        except:
            exponent=DualNum(exponent,0)
            return DualNum(exponent.val ** self.val, np.exp(self.val * np.log(exponent.val)) * (self.der * np.log(exponent.val) + (self.val / exponent.val) * exponent.der))

        
    # Overload equal
    def __eq__(self, other):
        '''
        Perform equality comparison of DualNum object
        
        Attributes
        ----------------------------------
        other: int or float
          value to compare with self, self is a DualNum object

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        
        '''
        try:
            output = (self.val == other.val) and np.array_equal(self.der, other.der)
        except AttributeError:
            # output is false because scalars are not equal to variables
            output = False
        return output

    # Overload not equal
    def __ne__(self, other):
        '''
        Perform inequality comparison of DualNum object
        
        Attributes
        ----------------------------------
        other: int or float
          value to compare with self, self is a DualNum object

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        
        '''
        return not self.__eq__(other)

    # Overload sin
    @staticmethod  
    def sin(other):
        """
        sine of other

        Attributes
        ----------------------------------
        other: DualNum
          argument of sine function

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
        try:
            return DualNum(np.sin(other.val), np.cos(other.val)*other.der)
        except:
            other=DualNum(other,0)
            return DualNum(np.sin(other.val), np.cos(other.val)*other.der)

    # Overload cos
    @staticmethod
    def cos(other):
        """
        cosine of other

        Attributes
        ----------------------------------
        other: DualNum
          argument of cosine function

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
        try:
            return DualNum(np.cos(other.val), -np.sin(other.val)*other.der)
        except:
            other=DualNum(other,0)
            return DualNum(np.cos(other.val), -np.sin(other.val)*other.der)
        
    # Overload tan
    @staticmethod
    def tan(other):
        """
        tangent of other

        Attributes
        ----------------------------------
        other: DualNum
          argument of tangent function

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
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

    # Overload exp
    @staticmethod
    def exp(other, base=np.e):
        """
        exponential of other

        Attributes
        ----------------------------------
        other: DualNum
          argument of exponential function
        base: float
          base of exponential function

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
        try:
            return DualNum(base**other.val, base**other.val*other.der)
        except:
            other=DualNum(other,0)
            return DualNum(base**other.val, base**other.val*other.der)

    # Overload log
    @staticmethod
    def log(other, base=np.e):
        """
        logarithm of other

        Attributes
        ----------------------------------
        other: DualNum
          argument of logarithm function
        base: float
          base of logarithm function

        Returns
        ---------------------------------
          DualNum object with updated value and derivative, parents of this new node is other
        """
        try:
            return DualNum(np.log(other.val)/np.log(base), 1/other.val*other.der/np.log(base))
        except:
            other=DualNum(other,0)
            return DualNum(np.log(other.val)/np.log(base), 1/other.val*other.der/np.log(base))
    
    # Overload logistic
    @staticmethod
    def logistic(other):
        """
        logistic of other

        Attributes
        ----------------------------------
        other: DualNum
          argument of logistic function

        Returns
        ---------------------------------
          DualNum object with updated value and derivative, parents of this new node is other
        """
        return 1/(1+DualNum.exp(-other))

    # Overload arcsin
    @staticmethod
    def arcsin(other):
        """
        arcsine of other

        Attributes
        ----------------------------------
        other: DualNum
          argument of arcsin function

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
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
        """
        arccos of other

        Attributes
        ----------------------------------
        other: DualNum
          argument of arccos function

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
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
        """
        arctan of other

        Attributes
        ----------------------------------
        other: DualNum
          argument of arctan function

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
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
        """
        sinh of other

        Attributes
        ----------------------------------
        other: DualNum
          argument of sinh function

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
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
        """
        cosh of other

        Attributes
        ----------------------------------
        other: DualNum
          argument of cosh function

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
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
        """
        tanh of other

        Attributes
        ----------------------------------
        other: DualNum
          argument of tanh function

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
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
        """
        sqrt of other

        Attributes
        ----------------------------------
        other: DualNum
          argument of sqrt function

        Returns
        ---------------------------------
          DualNum object with updated value and derivative
        """
        if isinstance(other, DualNum):
            new_other = np.sqrt(other.val)
            new_der = 0.5 / np.sqrt(other.val) * other.der
            return DualNum(new_other, new_der)
        elif other < 0:
            raise ValueError('Cannot take square roots of negative values')
        else:
            return np.sqrt(other)


class DualNumVec(DualNum):
    '''
    Use DualNumVec class for vector valued functions

    Attributes
    ------------------
    vals: list of DualNum objects
    entries of list are entries of the function
    jacobian: dict
    jacobian of the function

    Methods
    -----------------
    getvals
    computes forward pass of forward mode of automatic differentiation for each entry in vector function

    getgrad
    get entry in Jacobian of function, labeled by the entry of the function and the variable against which it is derived

    Examples
    ========
    Vector Input
    >>> seed=[1,0]
    >>> x1=DualNum(1,seed[0])
    >>> x2=DualNum(2,seed[1])
    >>> y=DualNumVec([x1*x2+DualNum.exp(x1*x2),x1**2+x2**2]) #[f1, f2]
    >>> print(y.getvals(),y.getgrad())
    >>> print("values are supposed to be :" + str(2+np.exp(2)) + " " + str(5))
    >>> print("grad is supposed to be : " + str(2+2*np.exp(2)) + " " + str(2))
    [9.389056098930649, 5] [16.778112197861297, 2.0]
    values are supposed to be :9.38905609893065 5
    grad is supposed to be : 16.7781121978613 2
    
    
    # Automized --> Avoid manually indexing into the seed vector

    >>> grad=np.zeros((3,2))
    >>> for i in range(3):
    >>>     seed=np.eye(1, 3, i)[0]
    >>>     x1=DualNum(1,seed[0])
    >>>     x2=DualNum(2,seed[1])
    >>> y=DualNumVec([x1*x2+DualNum.exp(x1*x2), x1+x2**2,x1/x2+15])
    >>> grad[:,i:i+1]=np.array([y.getgrad()]).T
    >>> print(grad)
    [[16.7781122  8.3890561]
     [ 1.         4.       ]
     [ 0.5       -0.25     ]]
    '''
    def __init__(self, vec):
        """
        Constructs the necessary attributes of the class

        Parameters
        -------------------
        vals: list of DualNum objects
          entries of vec are entries of the function
        """
        self.vec=vec
        self.grad=[]
        self.vals=[]
        
    def getgrad(self):
        """
        get entry in Jacobian of function, labeled by the entry of the function and the variable against which it is derived

        Output
        ------------------
        value of the entry of interest in the Jacobian 

        """
        for i,x in enumerate(self.vec):
            self.grad.append(x.der)
        return self.grad
    
    def getvals(self):
        """
        Computes forward pass of forward mode of automatic differentiation for each entry in vector function
        Updates the attribute jacobian of the class
        """
        for i,x in enumerate(self.vec):
            self.vals.append(x.val)
        return self.vals


class Node:
  """
  Basic building block to use reverse mode of automatic differentiation
  In a function, all variables/numbers and the results of operations on them are represented as Nodes
  Together they form a tree structure

  Attributes
  -----------------------------------
  val: int
    value of the variable/number
  parent1, panrent2: Node
    Current Node is the result of an operation on previous Node(s), called the parents
  der: dict
    local partial derivatives of the Nodes in function of their parent(s)
  
  Methods
  ----------------------------------
  __add__, __radd__, __mul__, __rmul__, __sub__, __rsub__, __neg__, __pow__, __rpow__, __truediv__, __rtruediv__
  dunder methods overloaded to operate on Node objects

  exp, sin, cos, ln, tan, cosh, sinh, tanh, logistic
  basic functions overloaded to be used with Node objects

  reverse
  computes the reverse pass, i.e. goes through the tree in reverse and computes the gradient of the function

  getgrad(self, var)
  outputs the value of the gradient of the function with respect to the variable 'var'
  """
  def __init__(self,val, parent1=None, parent2=None, der={}):
    """
    Constructs the necessary attributes for Node
    Attributes
    -----------------------------------
    val: int
     value of the variable/number
    parent1, panrent2: Node
      Current Node is the result of an operation on previous Node(s), called the parents
    der: dict
      local partial derivatives of the Nodes in function of their parent(s)   
    """
    self.val=val
    self.parent1=parent1
    self.parent2=parent2
    self.der=der
    self.grad={}

  #Overload add
  def __add__(self,other):
    """
    add two Nodes together

    Attributes
    ----------------------------------
    other: Node
      Node object to add self to

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node are self and other
    """
    try:
      return Node(self.val+other.val, parent1=self, parent2=other, der={"1": 1, "2": 1})
    except:
      aux=Node(other)
      return Node(self.val+aux.val,parent1=self, parent2=aux, der={"1": 1, "2": 1})
  
  # Overload radd
  def __radd__(self, other):
    """
    add two Nodes together

    Attributes
    ----------------------------------
    other: Node
      Node object to add self to

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node are self and other
    """
    return self.__add__(other)
  
  #Overload mul
  def __mul__(self,other):
    """
    multiply two Nodes together

    Attributes
    ----------------------------------
    other: Node
      Node object to mutliply self to

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node are self and other
    """
    try:
      return Node(self.val*other.val, parent1=self, parent2=other, der={"1":other.val, "2":self.val})
    except:
      aux=Node(other)
      return Node(self.val*aux.val, parent1=self, parent2=aux, der={"1":aux.val, "2":self.val})

  #Overload rmul
  def __rmul__(self,other):
    """
    multiply two Nodes together

    Attributes
    ----------------------------------
    other: Node
      Node object to mutliply self to

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node are self and other
    """
    return self.__mul__(other)
  
  #Overload substraction
  def __sub__(self,other):
    """
    subtract two Nodes together

    Attributes
    ----------------------------------
    other: Node
      Node object to substract self to

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node are self and other
    """
    try:
      aux=Node(-other.val,parent1=other.parent1, parent2=other.parent2, der=other.der)
      return self.__add__(aux)
    except:
      return self.__add__(Node(-other))

  #Overload rsub
  def __rsub__(self,other):
    """
    substract two Nodes together

    Attributes
    ----------------------------------
    other: Node
      Node object to substract self to

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node are self and other
    """
    aux=Node(-self.val,parent1=self.parent1, parent2=self.parent2, der=self.der)
    return aux.__add__(other)
  
  #Overload negation
  def __neg__(self):
    """
    negation of Node value

    Returns
    ---------------------------------
      Node object with updated value
    """
    return Node(-self.val,parent1=self.parent1, parent2=self.parent2, der=self.der)

  #Overload power
  def __pow__(self, exponent):
    """
    takes power of first Node to the other

    Attributes
    ----------------------------------
    exponent: Node
      Node object to be exponentiated with

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node are self and other
    """
    assert self.val>0, 'cannot have negative value for x in x**y as encounter log(x) in derivative'
    try:
      return Node(self.val**exponent.val, parent1=self, parent2=exponent, der={"1": exponent.val*self.val**(exponent.val-1), "2": self.val**exponent.val*np.log(self.val)})
    except:
      aux=Node(exponent)
      return Node(self.val**aux.val, parent1=self, parent2=aux, der={"1": aux.val*self.val**(aux.val-1), "2": self.val**aux.val*np.log(self.val)})

  #Overload rpow
  def __rpow__(self,other):
    """
    takes power of second Node to the other

    Attributes
    ----------------------------------
    other: Node
      Node object to be exponentiated with

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node are self and other
    """
    try: 
      test=other.val
    except:
      aux=Node(other)
      return aux.__pow__(self)
    else:
      return other.__pow__(self)

  #Overload division
  def __truediv__(self, other):
    """
    divide self with other

    Attributes
    ----------------------------------
    other: Node
      Node object to divide self with

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node are self and other
    """
    try: 
      return Node(self.val/other.val, parent1=self, parent2=other, der={"1": 1/other.val, "2": -self.val/(other.val)**2})
    except:
      aux=Node(other)
      return Node(self.val/aux.val, parent1=self, parent2=aux, der={"1": 1/aux.val, "2": -self.val/(aux.val)**2})

  #Overload rtruediv
  def __rtruediv__(self, other):
    """
    divide other with self

    Attributes
    ----------------------------------
    other: Node
      Node object to be divided by self

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node are self and other
    """
    try: 
      test=other.val
    except:
      aux=Node(other)
      return aux.__truediv__(self)
    else:
      return other.__truediv__(self)
  
  @staticmethod
  def exp(other, base=np.e):
    """
    exponential of other

    Attributes
    ----------------------------------
    other: Node
      argument of exponential function
    base: float
      base of exponential function

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node is other
    """
    try:
      return Node(base**other.val, parent1=other, parent2=None, der={"1": base**other.val })
    except:
      return base**other
  
  @staticmethod
  def sin(other):
    """
    sine of other

    Attributes
    ----------------------------------
    other: Node
      argument of sine function

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node is other
    """
    try:
      return Node(np.sin(other.val), parent1=other, der={"1": np.cos(other.val)})
    except:
      return np.sin(other)
  
  @staticmethod
  def cos(other):
    """
    cosine of other

    Attributes
    ----------------------------------
    other: Node
      argument of cosine function

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node is other
    """
    try:
      return Node(np.cos(other.val), parent1=other, der={"1": -np.sin(other.val)})
    except:
      return np.cos(other)

  @staticmethod
  def log(other, base=np.e):
    """
    logarithm of other

    Attributes
    ----------------------------------
    other: Node
      argument of logarithm function
    base: float
      base of logarithm function

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node is other
    """
    try: 
      return Node(np.log(other.val)/np.log(base), parent1=other, der={"1": 1/other.val/np.log(base)})
    except:
      return np.log(other)/np.log(base)

  @staticmethod
  def arcsin(other):
    """
    arcsine of other

    Attributes
    ----------------------------------
    other: Node
      argument of arcsin function

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node is other
    """
    try:
      if other.val > 1 or other.val <-1:
        raise ValueError('please use value between -1 and 1, inclusive')
      else:
        new_other = np.arcsin(other.val)
        new_der = 1 / np.sqrt(1 - other.val**2)
        
        arcsin = Node(new_other, parent1=other, der={"1": new_der})
      return arcsin
    except AttributeError:
      return np.arcsin(other)

  @staticmethod
  def arccos(other):
    """
    arccos of other

    Attributes
    ----------------------------------
    other: Node
      argument of arccos function

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node is other
    """
    try:
      if other.val > 1 or other.val <-1:
        raise ValueError('please use value between -1 and 1, inclusive')
      else:
        new_other = np.arccos(other.val)
        new_der =  -1 / np.sqrt(1 - other.val**2)
        
        arcsin = Node(new_other, parent1=other, der={"1": new_der})
      return arcsin
    except AttributeError:
      return np.arcsin(other)

  @staticmethod
  def arctan(other):
    """
    arctan of other

    Attributes
    ----------------------------------
    other: Node
      argument of arctan function

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node is other
    """
    try:
      new_other = np.arctan(other.val)
      new_der = 1 / (1 + np.power(other.val, 2))
        
      arctan = Node(new_other, parent1=other, der={"1": new_der})
      return arctan
    except AttributeError:
      return np.arctan(other)

  @staticmethod
  def sinh(other):
    """
    sinh of other

    Attributes
    ----------------------------------
    other: Node
      argument of sinh function

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node is other
    """
    return (Node.exp(other) - Node.exp(-other))/2

  @staticmethod
  def cosh(other):
    """
    cosh of other

    Attributes
    ----------------------------------
    other: Node
      argument of cosh function

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node is other
    """
    return (Node.exp(other) + Node.exp(-other))/2

  @staticmethod
  def tanh(other):
    """
    tanh of other

    Attributes
    ----------------------------------
    other: Node
      argument of tanh function

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node is other
    """
    return Node.sinh(other)/Node.cosh(other)

  @staticmethod
  def logistic(other):
    """
    logistic of other

    Attributes
    ----------------------------------
    other: Node
      argument of logistic function

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node is other
    """
    return 1/(1+Node.exp(-other))

  @staticmethod
  def sqrt(other):
    """
    sqrt of other

    Attributes
    ----------------------------------
    other: Node
      argument of sqrt function

    Returns
    ---------------------------------
      Node object with updated value and derivative, parents of this new node is other
    """
    if other.val < 0:
        raise ValueError('Cannot take square roots of negative values')
    return other**(1/2)




  def reverse(self):
    """
    computes the reverse pass, i.e. goes through the tree in reverse and computes the gradient of the function

    Return
    ---------------------
    grad: dict
    the updated gradient of the function, entries are labelled by variables
    """
    grad=self.grad
    def _in(node, sofar):
      """
      inner function called recursively to go through the tree

      Attributes:
      -------------------
      node: Node
        Node visited
      sofar: float
        value of the partial derivative computed so far

      Output
      -----------------
      None
      """
      par1=node.parent1
      par2=node.parent2
      der=node.der
      # print('-in is running')
      if par2!=None:
        try:
          grad[par1]+=sofar*der["1"]
        except:
          grad[par1]=sofar*der["1"]
        _in(par1,sofar*der["1"])
        try:
          grad[par2]+=sofar*der["2"]
        except:
          grad[par2]=sofar*der["2"]
        _in(par2,sofar*der["2"])

      if par1!=None and par2==None:
        try:
          grad[par1]+=sofar*der["1"]
        except:
          grad[par1]=sofar*der["1"]
        _in(par1,sofar*der["1"])

    _in(self, 1)
    return grad

  def getgrad(self,var):
    """
    Outputs value of gradient of the function with respect to var

    Attributes
    ---------------------------
    var: Node
      variable to compute gradient of function with respect to

    Output
    ---------------------------
    if reverse function has already been executed, outputs value of gradient with resepct to variable
    if reverse function not executed, KeyError

    """
    try: 
      return self.grad[var]
    except:
      raise KeyError('this is not a valid variable')



class NodeVec(Node):
  """
  Use Node class for vector valued functions

  Attributes
  ------------------
  vals: list of Node objects
    entries of list are entries of the function
  jacobian: dict
    jacobian of the function

  Methods
  -----------------
  reverse
    computes reverse pass of reverse mode of automatic differentiation for each entry in vector function

  getgrad
    get entry in Jacobian of function, labeled by the entry of the function and the variable against which it is derived
  """
  def __init__(self,vals):
    """
    Constructs the necessary attributes of the class

    Parameters
    -------------------
    vals: list of Node objects
      entries of list are entries of the function
    """
    self.vals=vals
    self.jacobian={}

  def reverse(self):
    """
    Computes reverse pass of reverse mode of automatic differentiation for each entry in vector function
    Updates the attribute jacobian of the class
    """
    for i,x in enumerate(self.vals):
      self.vals[i].reverse()
      self.jacobian[i]=self.vals[i].grad

  def getgrad(self, idx, var):
    """
    get entry in Jacobian of function, labeled by the entry of the function and the variable against which it is derived

    Parameters
    -----------------
    idx: int
      index of the function's entry

    var: Node
      variable against which function's entry is derived

    Output
    ------------------
    value of the entry of interest in the Jacobian 

    """
    if idx>0 and idx<=len(self.vals):
      try: 
        dic=self.jacobian[idx-1]
        return dic[var]
      except:
        raise KeyError('the given variable is not valid')
    else:
      raise KeyError('the index is out of range')


'''
# Presentation Tests
#Scalar Input (2 DualNum Objects)
x1=DualNum(0, 1)
x2=DualNum(1, 1)
y=x1+x2
print(y.val, y.der)

# Scalar Input (DualNum + Scalar)
y2=x1 + 3
print(y2.val, y2.der)

# Applying a function to the object
x3=2*DualNum.cos(DualNum(0,1))
print(x3.val, x3.der)

# Vector Input
seed=[1,0]
x1=DualNum(1,seed[0])
x2=DualNum(2,seed[1])
y=DualNumVec([x1*x2+DualNum.exp(x1*x2),x1**2+x2**2]) #[f1, f2]
print(y.getvals(),y.getgrad())
print("values are supposed to be :" + str(2+np.exp(2)) + " " + str(5))
print("grad is supposed to be : " + str(2+2*np.exp(2)) + " " + str(2))

# Automized --> Avoid manually indexing into the seed vector

grad=np.zeros((3,2))
for i in range(3):
    seed=np.eye(1, 3, i)[0]
    x1=DualNum(1,seed[0])
    x2=DualNum(2,seed[1])
    y=DualNumVec([x1*x2+DualNum.exp(x1*x2), x1+x2**2,x1/x2+15])
    grad[:,i:i+1]=np.array([y.getgrad()]).T
print(grad)


#Test case for scalar function as in lecture 11 slide 31
x1=Node(1)
x2=Node(2)
y=x1*x2+Node.exp(x1*x2)
print(y.val)
y.reverse()
print(y.grad)
y.getgrad(x2)



# #Test case for vector function
x1=Node(2)
x2=Node(3)
y1=4*x1*Node.cos(x2)*Node.exp(x1)
y2=5*x1**2/x2+2+3**x1
f=NodeVec([y1,y2])
f.reverse()
print("Our Aut Diff Reverse Mode")
print(f.getgrad(1,x1), f.getgrad(1,x2))
print(f.getgrad(2,x1), f.getgrad(2,x2))

print("---------")
print("Wolfram Alpha")
print(12*np.exp(2)*np.cos(3), -8*np.exp(2)*np.sin(3))
print(20/3+9*np.log(3), -20/9)
'''

