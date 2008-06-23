#!/usr/bin/python

import instant
instant.USE_CACHE = True
instant.VERBOSE = 0
from instant import create_extension, find_module_by_signature, import_module_by_signature

sig = "((instant unittest test16.py))"

if not find_module_by_signature(sig):
    print "Defining code"
    c_code = """
    class Sum { 
    public: 
      virtual double sum(double a, double b){
        return a+b;
      }
    };


    double use_Sum(Sum& sum, double a, double b) {  
      return sum.sum(a,b); 
    }
    """
    print "Compiling code"
    res = create_extension(code=c_code, signature=sig)
    print "res = ", res

print "Importing code"
newmodule = import_module_by_signature(sig)
Sum = newmodule.Sum
use_Sum = newmodule.use_Sum

sum = Sum()
a = 3.7
b = 4.8
c = use_Sum(sum,a,b)
print "The sum of %g and %g is %g"% (a,b,c) 


class Sub(Sum): 
  def __init__(self): 
    Sum.__init__(self) 
    
  def sum(self,a,b): 
    print "sub" 
    return a-b; 



sub = Sub()
a = 3.7
b = 4.8
c = use_Sum(sub,a,b)
print "The sub of %g and %g is %g"% (a,b,c) 





