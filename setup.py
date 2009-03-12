#!/usr/bin/env python

import sys, platform
from os.path import join, splitext
from distutils.core import setup

scripts = [join("etc", "instant-clean"), join("etc", "instant-showcache")]
if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts 
    # without the .py extension. A solution is to create batch files
    # that runs the different scripts.

    # try to determine the installation prefix
    # first set up a default prefix:
    if platform.system() == "Windows":
        prefix = sys.prefix
    else:
        # we are running bdist_wininst on a non-Windows platform
        pymajor, pyminor = sysconfig.get_python_version().split(".")
        prefix = "C:\\Python%s%s" % (pymajor, pyminor)

    # if --prefix is specified we use this instead of the default:
    for arg in sys.argv:
        if "--prefix" in arg:
            prefix = arg.split("=")[1]
            break

    # create batch files for Windows:
    for batch_file in ["instant-clean.bat", "instant-showcache.bat"]:
        f = open(batch_file, "w")
        f.write("@python %s %%*" % \
                join(prefix, "Scripts", splitext(batch_file)[0]))
        f.close()
        scripts.append(batch_file)

setup(name = "instant", version = '0.9.6', 
      description = "Instant Inlining of C/C++ in Python", 
      author = "Magne Westlie and Kent-Andre Mardal and Martin Sandve Alnes", 
      author_email = "magnew@simula.no, kent-and@simula.no, martinal@simula.no", 
      url = "http://www.fenics.org/instant", 
      packages = ['instant'],
      package_dir = {'instant': 'src/instant'}, 
      scripts = scripts,
      data_files = [(join("share", "man", "man1"),
                     [join("doc", "man", "man1", "instant-clean.1.gz"),
                      join("doc", "man", "man1", "instant-showcache.1.gz")])]
      )

