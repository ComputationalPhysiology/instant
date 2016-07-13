#!/usr/bin/env python

from __future__ import print_function
import os
import pytest

print("Cleaning local test cache before tests.")
os.system("python clean.py")

# Run tests
pytest.main()
