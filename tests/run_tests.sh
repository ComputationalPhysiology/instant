#!/bin/sh 

echo Cleaning local test cache before tests.
./clean.sh

rm -f failed_tests

for file in test*.py; do
   echo "Running test $file ";
   python $file;
   if [ $? != 0 ]; then
     echo $file >> failed_tests
   fi
done

if [ -f failed_tests ]; then  
  echo
  echo The following tests failed:
  cat failed_tests
fi

