#!/usr/bin/env python

import numpy as N
import time
from instant import inline_with_numpy

func_str = "sin" 
c_code = """
void gridloop(int d, int* dims, double* a,
              int n, double* x, int m, double* y) {
  for (int i=0; i<n; i++) {  
      for (int j=0; j<m; j++) {  
          a[i*n +j] = %s(x[i] + y[j]);
      }
  }
}
""" % func_str 

n = 5000

a = N.zeros([n, n]) 
x = N.arange(0.0, n, 1.0)
y = N.arange(0.0, n, 1.0)

arrays = [['d', 'dims', 'a'], ['n', 'x'], ['m', 'y']]
grid_func = inline_with_numpy(c_code, arrays=arrays )


t1 = time.time()
grid_func(a,x,y)
t2 = time.time()
print 'With instant:',t2-t1,'seconds'


xv = x[:, N.newaxis]
yv = y[N.newaxis, :]
a2 = N.zeros([n, n]) 
t1 = time.time()
a2[:,:] = N.sin(xv + yv)
t2 = time.time()
print 'With Numeric:',t2-t1,'seconds'

d = a-a2 
d.shape = (n*n,) 

print 'The difference is ', max(d)




