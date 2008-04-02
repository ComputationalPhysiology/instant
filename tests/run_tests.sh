#!/bin/sh 

for file in *.py; do
   echo "running test $file ";
   python $file;
   if [ $? != 0 ]; then
     exit
   fi
done


