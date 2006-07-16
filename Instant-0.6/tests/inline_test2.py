
import Numeric 
import time
from Instant import inline_with_numeric

c_code = """
void add(int n1, double* array1, int n2, double* array2, int n3, double* array3){
  for (int i=0; i<n1; i++) {  
    array3[i] = array1[i] + array2[i]; 
  }
}
"""


add_func = inline_with_numeric(c_code, arrays = [['n1', 'array1'],['n2', 'array2'],['n3', 'array3']])

a = Numeric.arange(10000000); a = Numeric.sin(a)
b = Numeric.arange(10000000); b = Numeric.cos(b)
c = Numeric.arange(10000000); c = Numeric.cos(c)
d = Numeric.arange(10000000); d = Numeric.cos(d)


t1 = time.time() 
add_func(a,b,c)
t2 = time.time()
print 'With Instant:',t2-t1,'seconds'

t1 = time.time() 
Numeric.add(a,b,d)
t2 = time.time()
print 'Med numpy:   ',t2-t1,'seconds'

difference = abs(d - c) 
sum = reduce( lambda a,b: a+b, difference)  
print "The difference between the arrays computed by numpy and instant is " + str(sum) 



