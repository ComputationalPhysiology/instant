#!/usr/bin/python

import Numeric as N 
import time
from Instant import create_extension
import sys


c_code = """
void func(int n1, double* array1, int n2, double* array2){
  double a; 
  for (int i=0; i<n1; i++) {  
    a = array1[i]; 
    array2[i] = sin(a) + cos(a) + tan(a);  
  }
}
"""

create_extension(code=c_code, headers=["arrayobject.h"], cppargs='-g',
          include_dirs=[sys.prefix + "/include/python" + sys.version[:3] + "/Numeric"],
          init_code='import_array();', module='test5_ext', 
          arrays = [['n1', 'array1'],['n2', 'array2']])



seed = 10000000.0


a = N.arange(seed) 
t1 = time.time()
b = N.sin(a) + N.cos(a) + N.tan(a)   
t2 = time.time()
print "With NumPy: ", t2-t1, "seconds" 


from test5_ext import func

c = N.arange(seed)
t1 = time.time()
func(a,c)
t2 = time.time()
print "With Instant: ", t2-t1, "seconds" 


t1 = time.time()
d = N.sin(a)
d += N.cos(a)
d += N.tan(a) 
t2 = time.time()
print "With NumPy inplace aritmetic: ", t2-t1, "seconds" 



difference = abs(b - c) 
sum = reduce( lambda a,b: a+b, difference)  
print "The difference between the arrays computed by numpy and instant is " + str(sum) 












