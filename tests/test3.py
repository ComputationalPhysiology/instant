#!/usr/bin/env python

import instant
from instant import create_extension  

c_code = """
double sum(double a, double b){
  return a+b;
}
"""

create_extension(code=c_code, modulename='test3_ext', cppargs=['-pg', '-O3', '-g'])

from test3_ext import sum 
a = 3.7
b = 4.8
c = sum(a,b)
print "The sum of %g and %g is %g" % (a,b,c) 

