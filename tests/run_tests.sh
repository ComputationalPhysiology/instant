#!/bin/sh 

sh clean.sh 

for file in *.py; do
   echo "running test $file ";
   python $file;
   if [ $? != 0 ]; then
     exit 1
   fi
done


