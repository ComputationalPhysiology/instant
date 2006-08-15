#!/usr/bin/env python

from distutils.core import setup

setup(name="Instant", version='0.7', 
      description="Instant Inlining of C/C++ in Python", 
      author="Magne Westlie and Kent-Andre Mardal", 
      author_email ="magnew@simula.no, kent-and@simula.no", 
      url="http://pyinstant.sf.net", 
      package_dir={'': 'src' }, 
      py_modules=['Instant'])



