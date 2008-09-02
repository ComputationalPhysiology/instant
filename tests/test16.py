#!/usr/bin/python

import instant

from instant import create_extension, find_extension, import_extension

sig = "((instant unittest test16.py))"

if not find_extension(sig, cache_dir=None):
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
    create_extension(code=c_code, signature=sig, cache_dir="test_cache")

print "Importing code"
newmodule = import_extension(sig)
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
