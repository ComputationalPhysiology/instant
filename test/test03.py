from __future__ import print_function
import pytest
import instant
from instant import build_module

c_code = """
double sum(double a, double b){
  return a+b;
}
"""

def test_build_module():
    build_module(code=c_code, modulename='test3_ext',
                 cppargs=['-pg', '-O3', '-g'], lddargs=['-pg'])

    from test3_ext import sum
    a = 3.7
    b = 4.8
    c = sum(a, b)
    assert c == 8.5
