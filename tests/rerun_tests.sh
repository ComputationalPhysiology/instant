#!/bin/sh 

for file in test*.py; do
   echo "Running test $file ";
   python $file;
   if [ $? != 0 ]; then
     exit 1
   fi
done
