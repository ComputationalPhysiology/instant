#!/usr/bin/env python

from distutils.core import setup

setup(name="Instant", version='0.3', 
      description="Instant Inlining of C/C++ in Python", 
      author="Magne Westlie and Kent-Andre Mardal", 
      author_email ="magnew@simula.no, kent-and@simula.no", 
      url="http://sourceforge.net/projects/pyinstant", 
      package_dir={'': 'src' }, 
      py_modules=['Instant'])



