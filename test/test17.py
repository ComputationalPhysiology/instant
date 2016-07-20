from __future__ import print_function
import pytest
from time import time
from instant import build_module, import_module

def test_repeated_build():
    #_t = None
    def tic():
        global _t
        _t = -time()

    def toc(msg=""):
        t = time() + _t
        print("t = %f  (%s)" % (t, msg))
        return t

    c_code = """
double sum(double a, double b)
{
  return a+b;
}
    """

    # Time a few builds
    tic()
    module = build_module(code=c_code)
    t1 = toc("first build")

    tic()
    module = build_module(code=c_code)
    t2 = toc("second build")

    tic()
    module = build_module(code=c_code)
    t3 = toc("third build")

    assert t1 > t2
    assert t1 > t3
