#!/bin/sh

files=`find  ../../Instant-0.1 -type f| grep -v svn` 
#echo $files 
tar -cf Instant-0.1.tar $files  
gzip Instant-0.1.tar
ls -s Instant-0.1.tar.gz

