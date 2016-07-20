from __future__ import print_function
import pytest
import numpy
import time
from instant import inline_with_numpy

def test_inline_numpy1():

    # Example 1: two arrays, one in, one inout
    c_code = """
double sum (int x1, int y1, int z1, double* array1, int x2, double* array2){
  double tmp = 0.0;
  for (int i=0; i<x1; i++)
    for (int j=0; j<y1; j++)
      for (int k=0; k<z1; k++){
        tmp += array1[i*y1*z1 + j*z1 + k];
        array2[1] = 2;
    }
  return tmp;
}
    """

    sum_func = inline_with_numpy(c_code, arrays = [['x1', 'y1', 'z1', 'array1'],
                                                   ['x2', 'array2', 'in']],
                                 cache_dir="test_ex1_cache")

    a = numpy.arange(27000); a = numpy.sin(a)
    c = a.copy()
    a.shape = (30, 30, 30)

    #print('b = (1,1)')
    b = (1., 1.)
    sum1 = sum_func(a, b)
    #print(b, 'b not changed when list')

    b = numpy.ones(2)
    sum1 = sum_func(a, b)
    #print(b, 'b not changed when numpy array')
    #print(sum1)


def test_inline_numpy2():

    # Example 2: two array, both inout and of same size
    # Cannot avoid specifying all dimensions for both arrays
    c_code = """
double sum (int x1, int y1, double* array1, int x2, int y2, double* array2){
  double tmp = 0.0;
  for (int i=0; i<x1; i++)
    for (int j=0; j<y1; j++){
      tmp = array1[i*y1 + j];
      array1[i*y1 + j] = array2[i*y1 + j];
      array2[i*y1 + j] = tmp;
    }
  return tmp;
}
    """

    sum_func = inline_with_numpy(c_code, arrays = [['x1', 'y1', 'array1'],
                                                   ['x2', 'y2', 'array2']],
                                 cache_dir="test_ex2_cache")

    a = numpy.ones(4)
    a.shape = (2, 2)
    b = a.copy()
    a *= 2

    sum1 = sum_func(a, b)
    #print('a and b changed')
    #print(a)
    #print(b)


def test_inline_numpy3():
    # Example 3: two arrays, one in, one out
    c_code = """
void sum (int x1, int y1, double* array1, int xy2, double* array2){
  for (int i=0; i<x1; i++)
    for (int j=0; j<y1; j++)
      array2[i*y1 + j] = array1[i*y1 + j]*2;
}
    """

    sum_func = inline_with_numpy(c_code, arrays = [['x1', 'y1', 'array1', 'in'],
                                                   ['xy2', 'array2', 'out']],
                                 cache_dir="test_ex3_cache")

    a = numpy.ones(4)
    a.shape = (2, 2)
    a *= 2

    c = sum_func(a, a.size)
    c.shape = a.shape
    #print(c.shape)


def test_inline_numpy4():
    # Example 4: three arrays, one in, one inout, and one out
    c_code = """
void sum (int x1, int y1, long* array1, int x2, int* array2, int x3, double* array3){
  for (int i=0; i<x1; i++){
    array3[i] = 0;
    for (int j=0; j<y1; j++)
      array3[i] += array1[i*y1 + j]*array2[j];
  }
}
    """

    sum_func = inline_with_numpy(c_code, arrays = [['x1', 'y1', 'array1', 'in', 'long'],
                                               ['x2', 'array2', 'int'],
                                               ['x3', 'array3', 'out', 'double']],
                                 cache_dir="test_ex4_cache")

    a = numpy.arange(9)#, dtype='int32')
    a.shape = (3, 3)
    b = numpy.arange(3, dtype='int32')

    c = sum_func(a, b, b.size)
    #print(c)
    #print(numpy.dot(a, b))


def test_inline_numpy5():
    # Example 5: arrays with more than 3 dimensions, uses old typemaps, only doubles
    c_code = """
void sum (int m, int* mp, double* array1, int n, int* np, double* array2){
  int w = mp[0], x = mp[1], y = mp[2], z = mp[3];
  for (int i=0; i<w; i++)
    for (int j=0; j<x; j++)
      for (int k=0; k<y; k++)
        for (int l=0; l<z; l++){
          *array2 = *array1*2;
          array1++;
          array2++;
        }
}
    """

    sum_func = inline_with_numpy(c_code, arrays = [['m', 'mp', 'array1', 'multi'],
                                                   ['n', 'np', 'array2', 'multi'],],
                                 cache_dir="test_ex5_cache")

    a = numpy.arange(16, dtype='float64')
    a.shape = (2, 2, 2, 2)
    b = a.copy()*0

    sum_func(a, b)
    assert numpy.array(a*2 == b).all()
