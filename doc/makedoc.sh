#!/bin/sh -x

epydoc   -c white -o html_reference \
-u http://www.fenics.org/instant \
--html ../src/instant.py 

