import numpy as np
class Node:
  def __init__(self,val, parent1=None, parent2=None, der={}):
    self.val=val
    self.parent1=parent1
    self.parent2=parent2
    self.der=der
    self.grad={}

  #Overload add
  def __add__(self,other):
    try:
      return Node(self.val+other.val, parent1=self, parent2=other, der={"1": 1, "2": 1})
    except:
      aux=Node(other)
      return Node(self.val+aux.val,parent1=self, parent2=aux, der={"1": 1, "2": 1})
  
  # Overload radd
  def __radd__(self, other):
    return self.__add__(other)
  
  #Overload mul
  def __mul__(self,other):
    try:
      return Node(self.val*other.val, parent1=self, parent2=other, der={"1":other.val, "2":self.val})
    except:
      aux=Node(other)
      return Node(self.val*aux.val, parent1=self, parent2=aux, der={"1":aux.val, "2":self.val})

  #Overload rmul
  def __rmul__(self,other):
    return self.__mul__(other)
  
  #Overload substraction
  def __sub__(self,other):
    try:
      aux=Node(-other.val,parent1=other.parent1, parent2=other.parent2, der=other.der)
      return self.__add__(aux)
    except:
      return self.__add__(Node(-other))

  #Overload rsub
  def __rsub__(self,other):
    print('tried1')
    aux=Node(-self.val,parent1=self.parent1, parent2=self.parent2, der=self.der)
    return aux.__add__(other)
  
  #Overload negation
  def __neg__(self):
    return Node(-self.val,parent1=self.parent1, parent2=self.parent2, der=self.der)

  #Overload power
  def __pow__(self, exponent):
    assert self.val>0, 'cannot have negative value for x in x**y as encounter log(x) in derivative'
    try:
      return Node(self.val**exponent.val, parent1=self, parent2=exponent, der={"1": exponent.val*self.val**(exponent.val-1), "2": self.val**exponent.val*np.log(self.val)})
    except:
      aux=Node(exponent)
      return Node(self.val**aux.val, parent1=self, parent2=aux, der={"1": aux.val*self.val**(aux.val-1), "2": self.val**aux.val*np.log(self.val)})

  #Overload rpow
  def __rpow__(self,other):
    try: 
      test=other.val
    except:
      aux=Node(other)
      return aux.__pow__(self)
    else:
      return other.__pow__(self)

  #Overload division
  def __truediv__(self, other):
    try: 
      return Node(self.val/other.val, parent1=self, parent2=other, der={"1": 1/other.val, "2": -self.val/(other.val)**2})
    except:
      aux=Node(other)
      return Node(self.val/aux.val, parent1=self, parent2=aux, der={"1": 1/aux.val, "2": -self.val/(aux.val)**2})

  #Overload rtruediv
  def __rtruediv__(self, other):
    try: 
      test=other.val
    except:
      aux=Node(other)
      return aux.__truediv__(self)
    else:
      return other.__truediv__(self)
  
  @staticmethod
  def exp(other):
    try:
      return Node(np.exp(other.val), parent1=other, parent2=None, der={"1": np.exp(other.val) })
    except:
      return np.exp(other)
  
  @staticmethod
  def sin(other):
    try:
      return Node(np.sin(other.val), parent1=other, der={"1": np.cos(other.val)})
    except:
      return np.sin(other)
  
  @staticmethod
  def cos(other):
    try:
      return Node(np.cos(other.val), parent1=other, der={"1": -np.sin(other.val)})
    except:
      return np.cos(other)

  @staticmethod
  def ln(other):
    try: 
      return Node(np.log(other.val), parent1=other, der={"1": 1/other.val})
    except:
      return np.log(other)

  def reverse(self):
    grad=self.grad
    def _in(node, sofar):
      #node: node visited
      #sofar: partial derivative so far
      par1=node.parent1
      par2=node.parent2
      der=node.der
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
    try: 
      return self.grad[var]
    except:
      raise KeyError('this is not a valid variable')



class NodeVec(Node):
  def __init__(self,vals):
    self.vals=vals
    self.jacobian={}

  def reverse(self):
    for i,x in enumerate(self.vals):
      self.vals[i].reverse()
      self.jacobian[i]=self.vals[i].grad

  def getgrad(self, idx, var):
    if idx>0 and idx<=len(self.vals):
      try: 
        dic=self.jacobian[idx-1]
        return dic[var]
      except:
        raise KeyError('the given variable is not valid')
    else:
      raise KeyError('the index is out of range')


"""
#Test case for scalar function as in lecture 11 slide 31
x1=Node(1)
x2=Node(2)
y=x1*x2+Node.exp(x1*x2)
print(y.val)
y.reverse()
print(y.grad)
y.getgrad(x2)
"""

"""
#Test case for vector function
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
"""

