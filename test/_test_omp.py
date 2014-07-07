from __future__ import print_function
from instant import inline_with_numpy
from numpy import *
import time
import os

c_code = r"""
void compute(int n, double* x, 
             int m, double*y) {   
    
    if ( n != m ) {
        printf("n and m should be equal");  
        return; 
    }
    #pragma omp parallel
    {
    int id;
    id = omp_get_thread_num(); 
    printf("Thread %d\n", id);

    #pragma omp for
    for (int i=0; i<m; i++) {
      y[i] =  sin(x[i]) +  cos(x[i]) + 3.4*x[i];  
    }
    }
}
"""

c_code_scalar  = r"""
void compute_scalar(int n, double* x, 
             int m, double*y) {   
    
    if ( n != m ) {
        printf("n and m should be equal");  
        return; 
    }
    for (int i=0; i<m; i++) {
      y[i] =  sin(x[i]);  
    }
}
"""




N = 8000000

compute_func = inline_with_numpy(c_code, arrays = [['n', 'x'], ['m', 'y']], cppargs = ['-fopenmp'], lddargs=['-lgomp'], system_headers=["omp.h"])  

x = arange(0, 1, 1.0/N) 
y = arange(0, 1, 1.0/N) 
t1 = time.time()
t3 = time.clock()
compute_func(x, y)
t2 = time.time()
t4 = time.clock()
print('With instant and OpenMP', t4-t3, 'seconds process time')
print('With instant and OpenMP', t2-t1, 'seconds wall time')

compute_func_scalar = inline_with_numpy(c_code_scalar, arrays = [['n', 'x'], ['m', 'y']])  

x = arange(0, 1, 1.0/N) 
y = arange(0, 1, 1.0/N) 
t5 = time.time()
t7 = time.clock()
compute_func_scalar(x, y)
t6 = time.time()
t8 = time.clock()
print('With instant ', t8-t7, 'seconds process time')
print('With instant ', t6-t5, 'seconds wall time')

print("")
print('Speed-up ', (t6-t5)/(t2-t1), 'seconds wall time')





