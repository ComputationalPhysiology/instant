#!/bin/sh

rm Instant-0.4.tar.gz
files=`find  ../../Instant-0.4 -type f| grep -v svn` 
echo $files 
tar -cf Instant-0.4.tar $files  
gzip Instant-0.4.tar
ls -s Instant-0.4.tar.gz

