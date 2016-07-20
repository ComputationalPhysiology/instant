from __future__ import print_function
import pytest
from instant import build_module
import numpy as N
import sys
import time


c_code = """
/* add function for matrices with all safety checks removed ..*/
void add(int x1, int y1, double* array1,
         int x2, int y2, double* array2,
         int x3, int y3, double* array3){

  for (int i=0; i<x1; i++) {
    for (int j=0; j<y1; j++) {
      *array3 = *array1 + *array2;
      array3++;
      array2++;
      array1++;
    }
  }
}
"""

def test_build_module():

    # Guess arrayobject is either in sys.prefix or /usr/local

    test6_ext = build_module(code=c_code, system_headers=["numpy/arrayobject.h"], cppargs=['-g'],
                             include_dirs=[N.get_include()],
                             init_code='import_array();', modulename='test6_ext',
                             arrays = [['x1', 'y1', 'array1'],
                                       ['x2', 'y2', 'array2'],
                                       ['x3', 'y3', 'array3']])

    from test6_ext import add
    a = N.arange(4000000); a = N.sin(a); a.shape=(2000, 2000)
    b = N.arange(4000000); b = N.cos(b); b.shape=(2000, 2000)
    c = N.arange(4000000); c = N.cos(c); c.shape=(2000, 2000)
    d = N.arange(4000000); d = N.cos(d); d.shape=(2000, 2000)

    t1 = time.time()
    add(a, b, c)
    t2 = time.time()

    t3 = time.time()
    N.add(a, b, d)
    t4 = time.time()

    e = abs(d-c)
    e.shape=(4000000,)

    max_difference = max(e)
    assert abs(max_difference) < 1.0e-12
