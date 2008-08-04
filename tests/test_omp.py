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
      y[i] =  sin(x[i]);  
    }
    }
}
"""


N = 20000000
#compute_func = inline_with_numpy(c_code, arrays = [['n', 'x'], ['m', 'y']], cppargs = ['-fopenmp'], lddargs=['-fopenmp'], system_headers=["omp.h"])  
compute_func = inline_with_numpy(c_code, arrays = [['n', 'x'], ['m', 'y']], cppargs = ['-fopenmp'], lddargs=['-lgomp'], system_headers=["omp.h"])  

# It is not possible to change the environment variable while
# running the script, that is, the change is not seen when
# calling compute_func. It needs to be set before running 
# this test.
# time.time is wall time, time.clock is process(es?) time.
os.environ['OMP_NUM_THREADS'] = '2'
#print os.environ['OMP_NUM_THREADS']
x = arange(0, 1, 1.0/N) 
y = arange(0, 1, 1.0/N) 
t1 = time.time()
t3 = time.clock()
compute_func(x,y)
t2 = time.time()
t4 = time.clock()
print 'With instant and OpenMP', t4-t3, 'seconds process time'
print 'With instant and OpenMP', t2-t1, 'seconds wall time'

#os.environ['OMP_NUM_THREADS'] = '1'
#x = arange(0, 1, 1.0/N) 
#y = arange(0, 1, 1.0/N) 
#t1 = time.time()
#compute_func(x,y)
#t2 = time.time()
#print 'With instant and OpenMP (1 thread)',t2-t1,'seconds'






