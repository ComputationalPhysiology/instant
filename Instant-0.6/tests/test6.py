#!/usr/bin/python

from Instant import create_extension  


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

create_extension(code=c_code, module='test6_ext')

from test6_ext import * 
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





