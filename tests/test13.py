#!/usr/bin/env python

import instant 
use_cache = True

add_func = instant.inline("double add(double a, double b){ return a+b; }", use_cache=use_cache) 

print "The sum of 3 and 4.5 is ", add_func(3, 4.5)  



