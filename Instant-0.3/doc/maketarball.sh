#!/bin/sh

files=`find  ../../Instant-0.3 -type f| grep -v svn` 
#echo $files 
tar -cf Instant-0.3.tar $files  
gzip Instant-0.3.tar
ls -s Instant-0.3.tar.gz

