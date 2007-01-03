#!/bin/sh
""":"
exec python $0 ${1+"$@"}
"""#"


from Instant import create_extension 
import Numeric,sys

a = Numeric.arange(10000000)
a = Numeric.sin(a)
b = Numeric.arange(10000000)
b = Numeric.cos(b)



s = """
PyObject* add(PyObject* a_, PyObject* b_){
  /*
  various checks
  */ 
  PyArrayObject* a=(PyArrayObject*) a_;
  PyArrayObject* b=(PyArrayObject*) b_;

  int n = a->dimensions[0];

  int dims[1];
  dims[0] = n; 
  PyArrayObject* ret;
  ret = (PyArrayObject*) PyArray_FromDims(1, dims, PyArray_DOUBLE); 

  int i;
  double aj;
  double bj;
  double *retj; 
  for (i=0; i < n; i++) {
    retj = (double*)(ret->data+ret->strides[0]*i); 
    aj = *(double *)(a->data+ a->strides[0]*i);
    bj = *(double *)(b->data+ b->strides[0]*i);
    *retj = aj + bj; 
  }
return PyArray_Return(ret);
}
"""


create_extension(code=s, system_headers=["arrayobject.h"],
              include_dirs=[sys.prefix + "/include/python" + sys.version[:3] + "/Numeric"],
              init_code='import_array();', module="test4_ext"
              )


import time
import test4_ext 

t1 = time.time() 
d = test4_ext.add(a,b)
t2 = time.time()

print 'With Instant:',t2-t1,'seconds'

t1 = time.time() 
c = a+b
t2 = time.time()

print 'Med numpy:   ',t2-t1,'seconds'

difference = abs(c - d) 

sum = reduce( lambda a,b: a+b, difference)  
print "The difference between the arrays computed by numpy and instant is " + str(sum) 






