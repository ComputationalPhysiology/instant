#!/bin/sh

files=`find  ../../Instant-0.2 -type f| grep -v svn` 
#echo $files 
tar -cf Instant-0.2.tar $files  
gzip Instant-0.2.tar
ls -s Instant-0.2.tar.gz

