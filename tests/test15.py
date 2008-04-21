
def test(): 

    x = symbol('x')
    pi = 3.14


    f =  -3*x**7 - 2*x**3 + 2*exp(x**2) + x**12*(2*exp(2*x) - pi*sin(x)**((-1) + pi)*cos(x)) + 4*(pi**5 + x**5 + 5*pi*x**4 + 5*x*pi**4 + 10*pi**2*x**3 + 10*pi**3*x**2)*exp(123 + x - x**5 + 2*x**4) 
        
        

    a = []
    t0 = time.time()
    for i in range(0,1000): 
        xx = i/1000.0
        y = f.subs(x==xx)
        a.append(y)
        
    t1 = time.time()

    print "Elapsed time with Swiginac  ", t1-t0

    from instant import inline
    func = inline("double f(double x) { return %s; }" % f.printc()) 

    b = []
    t0 = time.time()
    for i in range(0,1000): 
        xx = i/1000.0
        y = func(xx)
        b.append(y)
    t1 = time.time()


    print "Elapsed time with Swiginac expression inlined", t1-t0

    sum = 0 
    for i in range(0, len(a)): 
        relative_diff = (a[i] - b[i])/(a[i] + b[i])

    print "relative diff", relative_diff 



if __name__ == '__main__': 
    try:  
        from swiginac import * 
        import time
        test()
    except: 
        print "You need Swiginac for this test"






