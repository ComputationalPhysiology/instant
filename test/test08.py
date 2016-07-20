from __future__ import print_function
import pytest
from instant import build_module

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

def test_build_module():
    test8_ext = build_module(code=c_code, modulename='test8_ext')

    from test8_ext import Sum, use_Sum
    sum = Sum()
    a = 3.7
    b = 4.8
    c = use_Sum(sum, a, b)
    assert c == 8.5


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
    assert c == 8.5
