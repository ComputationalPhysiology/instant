#!/usr/bin/python

from instant import create_extension  
import numpy
import sys
import time


c_code = """
/* add function for vectors with all safety checks removed ..*/ 
void add(int n1, double* array1, int n2, double* array2, int n3, double* array3){
  if ( n1 == n2 && n1 == n3 ) {
     for (int i=0; i<n1; i++) {  
        array3[i] = array1[i] + array2[i]; 
     }
  }
  else {
    printf("The arrays should have the same size."); 
  }
  
}
"""


test5_ext = create_extension(code=c_code, system_headers=["numpy/arrayobject.h"], cppargs='-pg',
          include_dirs=[numpy.get_include()],
          init_code='import_array();', modulename='test5_ext', 
          arrays = [['n1', 'array1'],['n2', 'array2'],['n3', 'array3']])

from test5_ext import add 
add = test5_ext.add
a = numpy.arange(10000000); a = numpy.sin(a)
b = numpy.arange(10000000); b = numpy.cos(b)
c = numpy.arange(10000000); c = numpy.cos(c)
d = numpy.arange(10000000); d = numpy.cos(d)



t1 = time.time() 
add(a,b,c)
t2 = time.time()
print 'With instant:',t2-t1,'seconds'

t1 = time.time() 
numpy.add(a,b,d)
t2 = time.time()
print 'With numpy:   ',t2-t1,'seconds'

difference = abs(d - c) 
sum = reduce( lambda a,b: a+b, difference)  
print "The difference between the arrays computed by numpy and instant is " + str(sum) 


