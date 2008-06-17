#!/bin/sh


rm -rf ../build
rm instant-0.9.4.tar.gz

cd ../tests 
sh clean.sh 

cd ../etc
files=`find  ../../instant -type f| grep -v hg` 
echo $files 
tar -cf instant-0.9.4.tar $files  
gzip instant-0.9.4.tar
ls -s instant-0.9.4.tar.gz

