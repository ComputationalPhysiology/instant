#!/bin/sh -x

epydoc -t Instant -n Instant -c blue -o html_reference \
-u http://pypde.simula.no/Instant \
--html Instant \
../src/Instant.py \

