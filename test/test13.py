#!/usr/bin/env python

from __future__ import print_function
import instant 

add_func = instant.inline("double add(double a, double b){ return a+b; }", cache_dir="test_cache")

print("The sum of 3 and 4.5 is ", add_func(3, 4.5))  



