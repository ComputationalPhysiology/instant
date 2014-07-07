#!/usr/bin/env python 

from __future__ import print_function
import os, sys, glob, shutil

print("Cleaning local test cache before tests.")
os.system("python clean.py")

failed_tests = []
for f in glob.glob("test*.py"):
    print("Running test %s" % f)
    failure = os.system("python " + f)
    if failure:
        failed_tests.append(f)

if failed_tests:
    print("\nThe following tests failed:")
    for f in failed_tests:
        print(f)
    sys.exit(len(failed_tests))
