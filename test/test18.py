from __future__ import print_function
import pytest
import time
from instant import build_module, import_module


def test_timing():
    #_t = None
    def tic():
        global _t
        _t = -time.time()

    def toc(msg=""):
        t = time.time() + _t
        print("t = %f  (%s)" % (t, msg))
        return t

    c_code = """
double sum(double a, double b)
{
  return a+b;
}
    """

    class Sig:
        def __init__(self, sig):
            self.sig = sig

        def signature(self):
            time.sleep(1.0)
            return self.sig

        def __hash__(self):
            time.sleep(0.5)
            return hash(self.sig)

        def __cmp__(self, other):
            if isinstance(other, Sig):
                return cmp(self.sig, other.sig)
            return -1

    sig = Sig("((test18.py signature))")
    cache_dir = "test_cache"

    # Time a few builds
    tic()
    module = build_module(code=c_code, signature=sig, cache_dir=cache_dir)
    assert module is not None
    t1 = toc("first build")

    tic()
    module = build_module(code=c_code, signature=sig, cache_dir=cache_dir)
    assert module is not None
    t2 = toc("second build")

    tic()
    module = build_module(code=c_code, signature=sig, cache_dir=cache_dir)
    assert module is not None
    t3 = toc("third build")

    # Time importing
    tic()
    module = import_module(sig, cache_dir)
    assert module is not None
    t4 = toc("first import")

    tic()
    module = import_module(sig, cache_dir)
    assert module is not None
    t5 = toc("second import")

    assert t1 > 1
    assert t2 < 1 and t2 > 0.4
    assert t3 < 1 and t3 > 0.4
    assert t4 < 1 and t4 > 0.4
    assert t5 < 1 and t5 > 0.4
