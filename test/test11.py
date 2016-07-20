from __future__ import print_function
import pytest
import numpy
import time
import instant
from instant import inline_with_numpy

def test_numpy_inline():

    c_code = """
double sum_of_some_func(int n1, double* array1){
  double tmp = 0.0;
  for (int i=0; i<n1; i++) {
      tmp += some_func(array1[i]);
  }
  return tmp;
}
    """

    some_func = inline_with_numpy(c_code, arrays=[['n1', 'array1']],
                                  local_headers=["./some_func.h"],
                                  libraries = ["m"])

    a = numpy.arange(10000000); a = numpy.sin(a)

    t1 = time.time()
    b = some_func(a)
    t2 = time.time()
    print('With instant:', t2-t1, 'seconds')

    t1 = time.time()
    c = sum(numpy.sin(a) + numpy.cos(a))
    t2 = time.time()
    print('With  numpy :', t2-t1, 'seconds')

    assert abs(c - b) < 1.0e-12
