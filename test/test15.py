#!/usr/bin/python

from __future__ import print_function
def test2():

    x = Symbol('x')
    pi = 3.14

    f =  -3*x**7 - 2*x**3 + 2*exp(x**2) + x**12*(2*exp(2*x) - pi*sin(x)**((-1) + pi)*cos(x)) + 4*(pi**5 + x**5 + 5*pi*x**4 + 5*x*pi**4 + 10*pi**2*x**3 + 10*pi**3*x**2)*exp(123 + x - x**5 + 2*x**4)

    a = []
    t0 = time.time()
    for i in range(0, 1000):
        xx = i/1000.0
        y = f.subs(x, xx)
        a.append(y)

    t1 = time.time()

    print("Elapsed time with sympy", t1-t0)


def test3():

    x = Symbol('x')
    pi = 3.14

    f = lambda x : -3*x**7 - 2*x**3 + 2*exp(x**2) + x**12*(2*exp(2*x) - pi*sin(x)**((-1) + pi)*cos(x))  \
      + 4*(pi**5 + x**5 + 5*pi*x**4 + 5*x*pi**4 + 10*pi**2*x**3 + 10*pi**3*x**2)*exp(123 + x - x**5 + 2*x**4)

    a = []
    t0 = time.time()
    for i in range(0, 1000):
        xx = i/1000.0
        y = f(xx)
        a.append(y)

    t1 = time.time()

    print("Elapsed time with lambda and math", t1-t0)


if __name__ == '__main__':
    import time
    try:
        from sympy import *
        test2()
    except:
        print("You need sympy for this test")
    try:
        from math import *
        test3()
    except:
        print("You need math for this test")
    try:
        from sympycore import *
        test2()
    except:
        print("You need sympycore for this test")
