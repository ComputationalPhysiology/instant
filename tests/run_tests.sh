#!/bin/sh 

#instant-clean
rm -rf test_cache
rm -rf *_ext

for file in *.py; do
   echo "running test $file ";
   if [ $file != "test_omp.py" ]; then
     python $file;
   fi
   if [ $? != 0 ]; then
     exit 1
   fi
done
