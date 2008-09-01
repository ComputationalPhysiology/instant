#!/usr/bin/env python

from os.path import join
from distutils.core import setup

setup(name = "instant", version = '0.9.4', 
      description = "Instant Inlining of C/C++ in Python", 
      author = "Magne Westlie and Kent-Andre Mardal and Martin Sandve Alnes", 
      author_email = "magnew@simula.no, kent-and@simula.no, martinal@simula.no", 
      url = "http://www.fenics.org/instant", 
      packages = ['instant'],
      package_dir = {'instant': 'src/instant'}, 
      scripts = [join("etc", "instant-clean")])

