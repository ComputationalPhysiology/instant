#!/bin/sh -x

epydoc -t Instant -n Instant -c blue -o doc/code \
-u http://pypde.simula.no/Instant \
--html Instant \
./src/Instant.py \

