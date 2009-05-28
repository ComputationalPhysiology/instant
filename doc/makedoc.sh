#!/bin/sh -x

epydoc -c white -o html_reference \
-v -u http://www.fenics.org/instant \
--html ../src/instant/*.py

