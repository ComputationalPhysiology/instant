#!/bin/sh -x

epydoc -t instant -n instant -c blue -o html_reference \
-u http://www.fenics.org/instant \
--html instant \
../src/instant.py \

