
import Numeric 
import time
from instant import inline_with_numeric

c_code = """
double sum (int n1, double* array1){
  double tmp = 0.0; 
  for (int i=0; i<n1; i++) {  
      tmp += array1[i]; 
  }
  return tmp; 
}
"""


sum_func = inline_with_numeric(c_code, arrays = [['n1', 'array1']])

a = Numeric.arange(10000000); a = Numeric.sin(a)

t1 = time.time()
sum1 = sum_func(a)
t2 = time.time()
print 'With instant:',t2-t1,'seconds'

t1 = time.time() 
sum2 =  Numeric.sum(a)
t2 = time.time()
print 'Med Numeric:   ',t2-t1,'seconds'

difference = abs(sum1 - sum2) 
print "The difference between the sums computed by numpy and instant is " + str(difference) 



