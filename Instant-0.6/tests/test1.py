
from Instant import inline



add_func = inline("double add(double a, double b){ return a+b; }") 
print "The sum of 3 and 4.5 is ", add_func(3, 4.5)  



