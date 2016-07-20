from __future__ import print_function
import sys, os, threading
from instant import inline

def compile_stuff(i):
    c_code = """
    double sum(int a, int b)
    { return a + b; }
    // Code number %d
    """ % (i % 2)
    func = inline(c_code, cache_dir="test_locking_cache")
    s = func(i, 3)
    assert s == (i+3)
    print("In run %d, result is %d as expected." % (i, s))

if __name__ == "__main__":
    cmd = sys.argv[0]
    args = sys.argv[1:]

    if args:
        print("Compiling with args =", args)
        i, = args
        compile_stuff(int(i))

    else:
        print("Starting new processes to compile.")
        def run(i):
            os.system("python %s %d" % (cmd, i))
        threads = []
        for i in range(10):
            threads.append( threading.Thread(target=run, args=(i,)) )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
