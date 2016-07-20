from __future__ import print_function
import pytest
from instant import build_module
import numpy, sys
from functools import reduce

a = numpy.arange(10000000)
a = numpy.sin(a)
b = numpy.arange(10000000)
b = numpy.cos(b)

s = """
PyObject* add(PyObject* a_, PyObject* b_){
  /*
  various checks
  */
  PyArrayObject* a=(PyArrayObject*) a_;
  PyArrayObject* b=(PyArrayObject*) b_;

  int n = a->dimensions[0];

  npy_intp dims[1];
  dims[0] = n;
  PyArrayObject* ret;
  ret = (PyArrayObject*) PyArray_SimpleNew(1, dims, PyArray_DOUBLE);

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

def test_build_module():

    test4_ext = build_module(code=s, system_headers=["numpy/arrayobject.h"],
                             include_dirs=[numpy.get_include()],
                             init_code='import_array();', modulename="test4_ext")

    import time

    t1 = time.time()
    d = test4_ext.add(a, b)
    t2 = time.time()

    print('With instant:', t2 - t1, 'seconds')

    t1 = time.time()
    c = a + b
    t2 = time.time()

    print('With numpy:   ', t2 - t1, 'seconds')

    difference = abs(c - d)
    sum = reduce( lambda a, b: a + b, difference)
    assert sum < 1.0e-12
