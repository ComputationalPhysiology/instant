from __future__ import print_function
import pytest
import numpy as N
import time
from instant import build_module
import sys
from functools import reduce


c_code = """
void func(int n1, double* array1, int n2, double* array2){
    double a;
    if ( n1 == n2 ) {
        for (int i=0; i<n1; i++) {
            a = array1[i];
            array2[i] = sin(a) + cos(a) + tan(a);
        }
    } else {
        printf("The arrays should have the same size.");
    }
}
"""


def test_module():
    # Guess arrayobject is either in sys.prefix or /usr/local

    test7_ext = build_module(code=c_code, system_headers=["numpy/arrayobject.h"], cppargs='-g',
                             include_dirs=[N.get_include()],
                             init_code='import_array();', modulename='test7_ext',
                             arrays = [['n1', 'array1'], ['n2', 'array2']])

    seed = 10000000.0

    a = N.arange(seed)
    t1 = time.time()
    b = N.sin(a) + N.cos(a) + N.tan(a)
    t2 = time.time()
    print("With NumPy: ", t2-t1, "seconds")

    from test7_ext import func
    c = N.arange(seed)
    t1 = time.time()
    func(a, c)
    t2 = time.time()
    print("With instant: ", t2-t1, "seconds")

    t1 = time.time()
    d = N.sin(a)
    d += N.cos(a)
    d += N.tan(a)
    t2 = time.time()
    print("With NumPy inplace aritmetic: ", t2-t1, "seconds")

    difference = abs(b - c)
    sum = reduce( lambda a, b: a+b, difference)
    assert abs(sum) < 1.0e-12
