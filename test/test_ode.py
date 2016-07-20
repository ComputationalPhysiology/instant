from __future__ import print_function
from instant import inline_with_numpy
from numpy import *
import time


def time_loop2(p, Q, A, B, dt, N, p0):
    p[0] = p0
    for i in range(1, N):
        p[i] = p[i-1] + dt*(B*Q[i] - A*p[i-1])

c_code = """
void time_loop(int n, double* p,
             int m, double* Q,
             double A, double B,
             double dt, int N, double p0)
{
    if ( n != m ) {
        printf("n and m should be equal");
        return;
    }
    if ( n != N ) {
        printf("n and N should be equal");
        return;
    }

    p[0] = p0;
    for (int i=1; i<n; i++) {
        p[i] = p[i-1] + dt*(B*Q[i] - A*p[i-1]);
    }
}
"""

N = 100000
time_loop = inline_with_numpy(c_code, arrays = [['n', 'p'], ['m', 'Q']], cppargs='-g', cache_dir="test_cache")
p = zeros(N)
T = 20.0
Q = sin(arange(0, T, T/N))+1

t1 = time.time()
time_loop(p, Q, 1.0, 1.0, 1.0/N, N, 1.0)
t2 = time.time()
print('With instant:', t2-t1, 'seconds')

p2 = zeros(N)
t1 = time.time()
time_loop2(p2, Q, 1.0, 1.0, 1.0/N, N, 1.0)
t2 = time.time()
print('With Python :', t2-t1, 'seconds')

print('The max difference between p and p2 is ', max(abs(p - p2)))


tt = arange(0, 1, 1.0/N)

a = """
try:
    import pylab
    pylab.plot(tt, p)
    pylab.plot(tt, Q)
    pylab.xlabel('time ')
    pylab.ylabel('pressure ')
    pylab.title('Pressure ')
    #pylab.savefig('pressure_plot')
    pylab.show()
except:
    print "To get a plot of the solution install pylab"
"""
