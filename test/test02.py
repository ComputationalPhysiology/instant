from __future__ import print_function
import pytest
import numpy
import time
from instant import inline_with_numpy

c_code = """
double sum (int n1, double* array1){
  double tmp = 0.0;
  for (int i=0; i<n1; i++) {
      tmp += array1[i];
  }
  return tmp;
}
"""

def test_compare_numpy():

    sum_func = inline_with_numpy(c_code, arrays = [['n1', 'array1']], cache_dir="test_cache")

    a = numpy.arange(10000000); a = numpy.sin(a)

    t1 = time.time()
    sum1 = sum_func(a)
    t2 = time.time()
    print('With instant:', t2 - t1, 'seconds')

    t1 = time.time()
    sum2 =  numpy.sum(a)
    t2 = time.time()
    print('With numpy:   ', t2-t1, 'seconds')

    difference = abs(sum1 - sum2)
    assert difference < 1.0e-12
