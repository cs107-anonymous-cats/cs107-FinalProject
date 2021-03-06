# Introduction

This package implements automatic differentiation. This is important for complex computational problems, including optimization. 

# Background 

Automatic Differentiation is a set of techniques that executes a sequence of elementary arithmetic operations (addition, subtraction, multiplication, division, etc.) and elementary functions. AD applies the chain rule to these operations to compute derivatives of arbitrary order, which is when the order is a real rational, irrational, or complex number. AD is different from symbolic and numerical differentiation because it is more code efficient, can output a single expression, and does not have round-off errors in the discretization process/cancellation. AD is also popular because it can compute partial derivatives of functions with many inputs/independent variables, which is important for gradient-based optimization. The two forms of AD are the forward mode, where the chain rule is applied from inside to outside the given function/expression, while reverse mode goes from outside to inside.

![Chain Rule](https://github.com/cs107-anonymous-cats/cs107-FinalProject/tree/main/docs/chain_rule.png)
Automatic Differentiation uses computational graphs, which are functional descriptions using nodes and edges to describe the given computation. Edges represent values like scalars, vectors, matrices, or tensors, while nodes represent functions whose inputs are the incoming edges and the outputs are the outcoming edges.Another feature of computational graphs are that they are directed, which allowsus to follow the order of the computation. For forward propagation computation, we compute the function from inside to outside. More specifically, both forward and reverse modes use the chain rule to calculate the gradients, and in forward mode, the gradients are computed in the same order as the function evaluation. 

The following is an example of a computational graph:

![Computational Graph](https://github.com/cs107-anonymous-cats/cs107-FinalProject/tree/main/docs/comp_graph.png)
Source: https://kailaix.github.io/ADCME.jl/latest/tu_whatis/


# How to Use DualNum

#### Installing the package

Download from git using:  

git clone https://github.com/cs107-anonymous-cats/cs107-FinalProject.git

#### Dependencies 

pip3 install -r requirements.txt

#### Testing the package

In */tests* directory, run the following commands:  

bash run_tests.sh pytest -v  

bash run_tests.sh pytest --cov=.  


#### Importing the package: 

from dualnum import DualNum

#### Examples:

For a scalar function: 

![](HowtoUse1_scalarV2.png)

For a vector function:
Note: For now vector functions are not suported. This functionality will be soon implemented and an example will be given here.

![](HowtoUse2_vector.png)

# Software Organization

#### Directory Structure
Will be laid out as follows: 
![](directory_structure.png)

#### Modules
**DualNum** module will contain the class definition and methods for our dual number structure and computes forward mode automatic differentiation.
 
#### Testing and Coverage
We currently use pytest for both testing and coverage on its own, but we will introduce a CI and CodeCov when it becomes available to us.

#### Package Distribution
As of now, our package is distributed as a downloadable GitHub repo (See download instructions in "How To Use DualNum".  

Package will be distributed with PyPI. We will create *pyproject.toml* and *setup.cfg* files and then use **build** to build and upload the project.

#### Package Framework
Since we don't have too many modules to work with and we aren't building any sort of application, a packaging framework won't be used to avoid overcomplicating development.

# Implementation 

1. Core data structure: 

We primarily used arrays when implementing the automatic differentiation for vector functions. We designed our own data structure in DualNum. This class object takes two input attributes: val and der. We developed several methods in the class, which will be outlined in the following sections. This class object can be used to calculate the value and derivative of a function. 


2. Classes to implement: 

We implemented 1 class: DualNumber. We overloaded all the basic operations, and created elemental functions as static methods. 

A snippet of the class structure is as follows:

![](dualnum_example.png)

3. Methods and name attributes: 

**Attributes:**

- `self.val`: stores the value of the DualNum object. In practice, this will be the value of a function 
- `self.der`: stores the derivative of the DualNum object. 

All the following methods are overwritten for members of the DualNum class: addition, multiplication, subtraction, division, power. In addition the following elementary functions are overwritten as well as staticmethods: exponential, logarithm, sine, cosine (tangent is sine over cosine). All of these methods also work with numbers which are upgraded to dual numbers when encountered as can be seen in the following example:
![](add_methodV2.png)

**Basic operations:**

- `__add__`: for addition.
- `__radd__`: for reverse addition.
- `__mul__`: for multiplication.
- `__rmul__`: for reverse multiplication.
- `__sub__`: for subtraction.
- `__rsub__`: for reverse subtraction.
- `__truediv__`: for division. 
- `__rtruedive__`: for reverse division. 
- `__neg__`: for negation.
- `__pow__`: for the power calculation.i.e. DualNum(1,1)^3. 
- `__rpow__`: for the reverse power calculation.i.e. 3^DualNum(1,1)

**Static methods:** for trigonometry operations as well as exponentials and ln operations. 

- `__sin__`
- `__cos__`
- `__exp__`
- `__ln__`

An example of the `__sin__` function is as follows:
![](sin.png)
 
4. External dependencies: 

We used **numpy** as the sole dependency package, because it is fast and offers powerful tools for mathematical operations. At this stage we do not require other dependencies such as simpy or scipy. 

5. Deal with elementary functions: 

We have desgined a **test_DualNum.py** file that achieved 96% of the code coverage. See below for the result. 

![](test_DualNum.png)

Some simple user cases for some methods: 

![](test_case.png)

# Future Features: 
We will implement forward mode for vector functions to handle multiple inputs as well as implementing reverse mode, which will be the main extension of our project. We chose the reverse mode as our extension because when the number of inputs is significantly greater than the number of outputs, the reverse mode will be computationally faster and efficient because fewer operations need to be performed. On the other hand, when the number of inputs is significantly smaller than the number outputs, the forward mode uses fewer computations to calculate the derivative. Thus, having both forward and reverse mode will offer our package more versatility. 

A challenge we will face is determining how to store and calculate partial derviatives, which is required in the reverse mode extension in the backward pass. Wewill most likely store partials as nodes in a common array, which is known as a tape in Automatic Differentiation literature. The tape will enable us to see each operation performed on the expression, which helps to compute the gradient when read in reverse. We will also probably create a new package for the reverse mode.

# Licensing

MIT License

Copyright (c) [2021] [Jie Sun, Taro Johann Spirig, Zachary Brown, Hannah Phan] 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# Feedback
More specifically, 
a) in the background section, it will be helpful to add some information about computational graphs and how they are used in AD. 

Implemented feedback: We included an explanation for the application of computational graphs in automatic differentiation, as well as embedded images for the chain rule and an example computational graph.

b) In "how to use" I would add some information about how this package can be downloaded/installed, and some more detailed demo examples of you creating some scalar/vector functions, and getting their values and derivatives. 

Implemented feedback: We explained how to install the package, import the package, and gave an example of the use case for forward mode (a scalar and vector function).

c) In implementation details, I would consider some of the elementary functions that you would add (like __add__, sin, __mul__, etc.), and have either a short description for all of them, or some basic pseudocode for one or two of them.

Implemented feedback: We provided examples of specific elementary functions, how we would implement dual numbers, and then some pseudocode for it. 
