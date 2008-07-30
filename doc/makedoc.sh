#!/bin/sh -x

epydoc   -c blue -o html_reference \
-u http://www.fenics.org/instant \
--html instant \
../src/instant.py \

