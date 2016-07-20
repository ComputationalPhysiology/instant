from __future__ import print_function
import pytest
import time
#from math import exp, sin, cos

def test_2():

    try:
        from sympy import Symbol, exp, sin, cos
    except:
        print("You need sympy for this test")
        return

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


def test_3():
    try:
        from sympy import Symbol, exp, sin, cos
    except:
        print("You need sympy for this test")
        return

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
