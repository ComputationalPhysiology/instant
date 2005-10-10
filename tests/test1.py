import Instant  

ext = Instant.Instant()

c_code = """
double sum(double a, double b){
  return a+b;
}
"""

ext.create_extension(code=c_code,
                     module='test1_ext')

from test1_ext import sum 
a = 3.7
b = 4.8
c = sum(a,b)
print "The sum of %g and %g is %g"% (a,b,c) 


