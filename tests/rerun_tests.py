#!/usr/bin/env python

import os, sys, glob

print "Not cleaning local test cache before tests."

failed_tests = []
for f in glob.glob("test*.py"):
   print "Running test", f
   failure = os.system("python " + f)
   if failure:
       failed_tests.append(f)

if failed_tests:
    print "\nThe following tests failed:"
    for f in failed_tests:
        print f
    sys.exit(len(failed_tests))
