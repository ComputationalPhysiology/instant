#!/bin/sh


rm -rf ../build
rm instant-0.9.5.tar.gz

cd ../tests 
sh clean.sh 

cd ../..
cp -r instant instant-0.9.5
cd instant-0.9.5/etc

files=`find  ../../instant-0.9.5 -type f| grep -v hg` 
#echo $files 
tar -cf instant-0.9.5.tar $files  
gzip instant-0.9.5.tar
ls -s instant-0.9.5.tar.gz
cp instant-0.9.5.tar.gz ../../instant/etc/. 
rm -rf ../../instant-0.9.5 


