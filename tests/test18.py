#!/usr/bin/python

import time
from instant import build_module, import_module

_t = None
def tic():
    global _t
    _t = -time.time()

def toc(msg=""):
    t = time.time() + _t
    print "t = %f  (%s)" % (t, msg)
    return t

c_code = """
double sum(double a, double b)
{
  return a+b;
}
"""

class Sig:
    def signature(self):
        time.sleep(1)
        return "((test18.py signature))"
sig = Sig()
cache_dir = "test_cache"

# Time a few builds
tic()
module = build_module(code=c_code, signature=sig, cache_dir=cache_dir)
t1 = toc("first build")

tic()
module = build_module(code=c_code, signature=sig, cache_dir=cache_dir)
t2 = toc("second build")

tic()
module = build_module(code=c_code, signature=sig, cache_dir=cache_dir)
t3 = toc("third build")

# Time importing
tic()
module = import_module(sig, cache_dir)
t4 = toc("first import")

tic()
module = import_module(sig, cache_dir)
t5 = toc("second import")

assert t1 > 1
assert t2 < 1
assert t3 < 1
assert t4 < 1
assert t5 < 1

