from __future__ import print_function
import numpy
import time
from instant import inline_module_with_numpy

c_code = """
double sum (int n1, double* array1){
  double tmp = 0.0;
  for (int i=0; i<n1; i++) {
      tmp += array1[i];
  }
  return tmp;
}
double max(int n1, double* array1){
  double tmp = 0.0;
  for (int i=0; i<n1; i++) {
      if (array1[i] > tmp) {
          tmp = array1[i];
      }
  }
  return tmp;
}

"""


c_module = inline_module_with_numpy(c_code, arrays = [['n1', 'array1']], cache_dir="test_cache")


a = numpy.arange(10000000); a = numpy.sin(a)

sum_func = c_module.sum
max_func = c_module.max


sum = sum_func(a)
max = max_func(a)

print("sum ", sum)
print("max ", max)
