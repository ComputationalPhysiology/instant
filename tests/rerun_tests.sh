#!/bin/sh 

echo Not cleaning local test cache before tests.
rm -f failed_tests

for file in test*.py; do
   echo "Running test $file ";
   python $file;
   if [ $? != 0 ]; then
     echo $file >> failed_tests
   fi
done

echo
echo The following tests failed:
cat failed_tests

