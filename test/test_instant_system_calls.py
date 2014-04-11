#!/usr/bin/env python

import os
from instant import inline

os.environ["INSTANT_SYSTEM_CALL_METHOD"] = "SUBPROCESS"
add_func = inline("double add(double a, double b){ return a+b; }", cache_dir="test_cache") 
print("The sum of 3 and 4.5 is ", add_func(3, 4.5))  

os.environ["INSTANT_SYSTEM_CALL_METHOD"] = "OS_SYSTEM"
add_func = inline("double add(double c, double d){ return c+d; }", cache_dir="test_cache") 
print("The sum of 3 and 4.5 is ", add_func(3, 4.5))  

os.environ["INSTANT_SYSTEM_CALL_METHOD"] = "COMMANDS"
add_func = inline("double add(double e, double f){ return e+f; }", cache_dir="test_cache") 
print("The sum of 3 and 4.5 is ", add_func(3, 4.5)) 
