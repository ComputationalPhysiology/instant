#!/usr/bin/env python

import numpy 
import time
from instant import inline_with_numpy

c_code = """
    double sum (int n1, double* array1){
        #pragma omp parallel default(none) shared(array1) private(i) reduction(+:tmp)
        {
            double tmp = 0.0; 
            #pragma omp for
            for (int i=0; i<n1; i++) {  
                tmp += array1[i]; 
            }
            return tmp; 
        }
    }
"""


sum_func = inline_with_numpy(c_code, arrays = [['n1', 'array1']])

a = numpy.arange(10000000); a = numpy.sin(a)

t1 = time.time()
sum1 = sum_func(a)
t2 = time.time()
print 'With instant:',t2-t1,'seconds'

t1 = time.time() 
sum2 =  numpy.sum(a)
t2 = time.time()
print 'With numpy:   ',t2-t1,'seconds'

difference = abs(sum1 - sum2) 
print "The difference between the sums computed by numpy and instant is " + str(difference) 
