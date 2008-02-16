#!/usr/bin/python

from instant import create_extension  
import Numeric as N
import sys
import time



c_code = """
/* add function for matrices with all safety checks removed ..*/ 
void add(int n1, int* p1, double* array1, 
         int n2, int* p2, double* array2, 
         int n3, int* p3, double* array3){

  for (int i=0; i<p1[0]; i++) {
    for (int j=0; j<p1[1]; j++) {
      *array3 = *array1 + *array2; 
      array3++; 
      array2++; 
      array1++; 
    }
  }
}
"""

create_extension(code=c_code, system_headers=["arrayobject.h"], cppargs='-g',
          include_dirs=[sys.prefix + "/include/python" 
                       + sys.version[:3] + "/Numeric"],
          init_code='import_array();', module='test6_ext', 
          arrays = [['n1', 'p1', 'array1'],
                    ['n2', 'p2', 'array2'],
                    ['n3', 'p3', 'array3']])

from test6_ext import add 
a = N.arange(4000000); a = N.sin(a); a.shape=(2000,2000)
b = N.arange(4000000); b = N.cos(b); b.shape=(2000,2000)
c = N.arange(4000000); c = N.cos(c); c.shape=(2000,2000)
d = N.arange(4000000); d = N.cos(d); d.shape=(2000,2000)

t1 = time.time() 
add(a,b,c)
t2 = time.time()

t3 = time.time() 
N.add(a,b,d)
t4 = time.time()

max_difference = max(max(abs(d - c))) 
print "The difference between the arrays computed by numpy and instant is ", max_difference 

print 'With instant:',t2-t1,'seconds'
print 'Med numpy:   ',t4-t3,'seconds'

