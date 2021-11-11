#### What tasks has each group member been assigned to for Milestone 2?
***

Our group met over the weekend to discuss the work breakdown structure and project extensions. We decided to add a reverse mode implementation for the extension.

__**Hannah**__: 

I will work on implementing the trignometric functions for single input scalar functions (add, subtract, negation, divide, multiply), implementing logarithmic functions (exp, pow, sqrt, etc.), and writing test cases for the functions that I did not write to ensure full coverage. I will also start on creating a skeleton that is compatible with multiple inputs, along with their test cases. Lastly, I will help on the reverse mode documentation. 

 
__**Jie**__:

Implementation details (documentation for forward and reverse mode implementation)

__**Taro**__:

I will work of the forward mode of automatic differentiation, specifically the dual numbers class and implementation of trignometric functions, as well as dealing with the case  when the value used in a function is a scalar versus a DualNum object. I will also work on writing test cases for the scalar functions.  

__**Zach**__: 

How to download the package, setting up PyPI, how to use the package, how to distribute and deliver the package, give use cases). software organization. Test suite (work with Hannah and Taro)


#### What has each group member done since the submission of Milestone 1?
***

Each member updated the milestone1 file based on the feedbacks. 

__**Hannah**__:
Since the submission of Milestone 1, I went to office hours to discuss the implementation of feedback with a TF and added graphics for the chain rule and computational graphs in the milestone 1 document and summarized everyone's implemented feedback for each section of the milestone. I also met with my groupmmates to discuss the work breakdown to complete milestone 2. I also started writing the function overloads for __add__, __radd__, __mul__, __rmul__, __neg__, __div__, __rdiv__, __pow___, and __rpow__.

__**Jie**__:

__**Taro**__:
Since the submission of Milestone 1, I went to office hours to discuss feedback implementation and I have updated the documentation according to the feedback. We also worked with Hannah on writing the overloading of trignometric functions and creating the DualNum class structure. I also met with the TF to discuss milestone 2 deliverables.

__**Zach**__:

Notes to the group/questions we discussed with Sehaj:
Qn: do we use codecov? yes we do not need to use TravisCI but we should take a screen shot of the results of our tests. Codecov will just look if we are testing most of the code. 90% coverage should be enoug.
Do we edit the intro and background. what goes into the how to use piece? what's in software organization? We will discuss that next week once we get feedback on Milestone 2A. What's in the future features? Probably reverse mode is a good idea.
