# Introduction

This package implements the automatic differentiation. This is important for complex computational problems, including optimization. 

# Background

Automatic Differentiation is a set of techniques that executes a sequence of elementary arithmetic operations (addition, subtraction, multiplication, division, etc.) and elementary functions. AD applies the chain rule to these operations to compute derivatives of arbitrary order, which is when the order is a real rational, irrational, or complex number. AD is different from symbolic and numerical differentiation because it is more code efficient, can output a single expression, and does not have round-off errors in the discretization process/cancellation. AD is also popular because it can compute partial derivatives of functions with many inputs/independent variables, which is important for gradient-based optimization. The two forms of AD are the forward mode, where the chain rule is applied from inside to outside the given function/expression, while reverse mode goes from outside to inside.

# How to Use AutomaticDifferentiation

The user should import the class AutomaticDifferentiation which has a forward and a reverse method. The user can then use those methods to compute the gradient of a function evaluated at a given point , i.e. AutomaticDifferentiation.forward(f,x) and AutomaticDifferentiation.reverse(f,x) with f the function whose gradient should be computed and x an array representing the point at which it should be evaluated.

# Software Orgnaization

- All source code along can be put into a /src directory, tests in /tests, and documentation 
in /docs. 
- We will have the module autodiff, which handles automatic differentiation.
- TravisCI and CodeCov will be used for testing.
- Package will be distributed by uploading to PyPI.
- We may or may not package our software using Django.

# Implementation

1. Core data structure: we will primarily use arrays. We will design our own data structure for dual numbers. 
2. Classes to implement: Differentation(). DualNumber().
3. Methods and name attributes: .forward, .reverse, primaltrace,tangenttrace. 
4. External dependencies: scipy, numpy.
5. Deal with elementary functions: we will overload those functions.

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
