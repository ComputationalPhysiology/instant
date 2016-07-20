from __future__ import print_function
import pytest
import instant
from instant import build_module, import_module

def test_sum():

    sig = "((instant unittest test16.py))"

    # Trying to import module
    module = import_module(sig, cache_dir="test_cache")
    if module is None:
        print("Defining code")
        c_code = """
        class Sum {
        public:
          virtual double sum(double a, double b){
          return a+b;
        }
      };

      double use_Sum(Sum& sum, double a, double b) {
        return sum.sum(a,b);
      }
        """
        print("Compiling code")
        module = build_module(code=c_code, signature=sig, cache_dir="test_cache")

    # Testing module
    Sum = module.Sum
    use_Sum = module.use_Sum

    sum = Sum()
    a = 3.7
    b = 4.8
    c = use_Sum(sum, a, b)
    print("The sum of %g and %g is %g"% (a, b, c))

    class Sub(Sum):
        def __init__(self):
            Sum.__init__(self)

        def sum(self, a, b):
            print("sub")
            return a-b;

    sub = Sub()
    a = 3.7
    b = 4.8
    c = use_Sum(sub, a, b)
    print("The sub of %g and %g is %g"% (a, b, c))
