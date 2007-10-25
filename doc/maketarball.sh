#!/bin/sh

cd ../tests 
rm -rf ../build
sh clean.sh 
cd ../doc 
rm instant-0.9.2.tar.gz
files=`find  ../../instant -type f| grep -v hg` 
echo $files 
tar -cf instant-0.9.2.tar $files  
gzip instant-0.9.2.tar
ls -s instant-0.9.2.tar.gz

