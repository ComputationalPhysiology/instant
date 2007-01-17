#!/usr/bin/env python

from distutils.core import setup

setup(name="instant", version='0.8', 
      description="Instant Inlining of C/C++ in Python", 
      author="Magne Westlie and Kent-Andre Mardal", 
      author_email ="magnew@simula.no, kent-and@simula.no", 
      url="http://www.fenics.org/instant", 
      package_dir={'': 'src' }, 
      py_modules=['instant'])



