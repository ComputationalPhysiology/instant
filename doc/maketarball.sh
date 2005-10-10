#!/bin/sh

files=`find  ../../Instant -type f| grep -v svn` 
#echo $files 
tar -cf Instant.tar $files  
gzip Instant.tar
ls -s Instant.tar.gz

