from __future__ import print_function
import pytest
import shutil
import time
import os, sys
from instant import build_module, import_module

def test_build():

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

    modulename = "test19_ext"
    cache_dir = "test19_cache"
    shutil.rmtree(cache_dir, ignore_errors=True)
    shutil.rmtree(modulename, ignore_errors=True)

    # Build and rebuild with explicit modulename
    tic()
    module = build_module(code=c_code, modulename=modulename,
                          cache_dir=cache_dir)
    assert module is not None
    t1 = toc("(1) With modulename")

    tic()
    module = build_module(code=c_code, modulename=modulename,
                          cache_dir=cache_dir)
    assert module is not None
    t2 = toc("(2) With modulename")
    assert t1 > t2

    # Try importing module in a separate python process
    python_interp = sys.executable
    cmd = python_interp + ' -c "import %s"' % modulename
    print(cmd)
    stat = os.system(cmd)
    assert stat == 0 # a

    # Build and rebuild with a valid filename as signature
    sig = "test19_signature_module"
    tic()
    module = build_module(code=c_code, signature=sig, cache_dir=cache_dir)
    assert module is not None
    t1 = toc("(1) With signature")

    tic()
    module = build_module(code=c_code, signature=sig, cache_dir=cache_dir)
    assert module is not None
    t2 = toc("(2) With signature")
    assert t1 > t2

    tic()
    module = import_module(sig, cache_dir)
    assert module is not None
    t3 = toc("(3) import_module")
    assert t1 > t3

    # Try importing module in a separate python process
    python_interp = sys.executable
    cmd = python_interp + ' -c "import instant; assert instant.import_module(\'%s\', \'%s\') is not None"' % (sig, cache_dir)
    print(cmd)
    stat = os.system(cmd)
    assert stat == 0 # b

    # Build and rebuild with generic signature string
    sig = "((test19_signature_module))"
    tic()
    module = build_module(code=c_code, signature=sig, cache_dir=cache_dir)
    assert module is not None
    t1 = toc("(1) With signature")

    tic()
    module = build_module(code=c_code, signature=sig, cache_dir=cache_dir)
    assert module is not None
    t2 = toc("(2) With signature")
    assert t1 > t2

    tic()
    module = import_module(sig, cache_dir)
    assert module is not None
    t3 = toc("(3) import_module")
    assert t1 > t3

    # Try importing module in a separate python process
    python_interp = sys.executable
    cmd = python_interp + ' -c "import instant; assert instant.import_module(\'%s\', \'%s\') is not None"' % (sig, cache_dir)
    print(cmd)
    stat = os.system(cmd)
    assert stat == 0 # c

    print("Skipping unit test, see https://bugs.launchpad.net/instant/+bug/518389")
    # Build and rebuild with generic signature object
    #sig = Sig("((test19_signature_module))")
    #tic()
    #module = build_module(code=c_code, signature=sig, cache_dir=cache_dir)
    #assert module is not None
    #t1 = toc("(1) With signature")
    #tic()
    #module = build_module(code=c_code, signature=sig, cache_dir=cache_dir)
    #assert module is not None
    #t2 = toc("(2) With signature")
    #assert t1 > t2
    #tic()
    #module = import_module(sig, cache_dir)
    #assert module is not None
    #t3 = toc("(3) import_module")
    #assert t1 > t3

    # Build and rebuild without modulename or signature
    tic()
    module = build_module(code=c_code, cache_dir=cache_dir)
    assert module is not None
    t1 = toc("(1) Without modulename or signature")

    tic()
    module = build_module(code=c_code, cache_dir=cache_dir)
    assert module is not None
    t2 = toc("(2) Without modulename or signature")
    assert t1 > t2
