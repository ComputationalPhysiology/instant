#!/bin/sh -x

epydoc -c white -o html \
-v -u http://www.fenics.org/instant \
--html ../src/instant/*.py

