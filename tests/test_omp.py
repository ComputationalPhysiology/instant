from instant import inline_with_numpy
from numpy import *
import time
import os

c_code = """
void compute(int n, double* x, 
             int m, double*y) {   
    if ( n != m ) {
        printf("n and m should be equal");  
        return; 
    }

   #pragma omp for
   for (int i=0; i<m; i++) {
     y[i] =  sin(x[i]);  
   }
}
"""


N = 100000        
compute_func = inline_with_numpy(c_code, arrays = [['n', 'x'], ['m', 'y']], cppargs = '-fopenmp', system_headers=["omp.h"], libraries=['gomp'])  
 
os.environ['OMP_NUM_THREADS'] = '2'
x = arange(0, 1, 1.0/N) 
y = arange(0, 1, 1.0/N) 
t1 = time.time()
compute_func(x,y)
t2 = time.time()
print 'With instant and OpenMP',t2-t1,'seconds'

os.environ['OMP_NUM_THREADS'] = '1'
x = arange(0, 1, 1.0/N) 
y = arange(0, 1, 1.0/N) 
t1 = time.time()
compute_func(x,y)
t2 = time.time()
print 'With instant and OpenMP',t2-t1,'seconds'






