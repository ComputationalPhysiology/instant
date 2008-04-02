#!/bin/sh 

for file in *.py;  
do 
echo "running test $file ";
python $file; 
done


