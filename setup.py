#!/usr/bin/env python

import sys, platform
from os.path import join, split, pardir

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.version_info < (2, 7):
    print("Python 2.7 or higher required, please upgrade.")
    sys.exit(1)

scripts = [join("scripts", "instant-clean"),
           join("scripts", "instant-showcache")]

if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        f = open(batch_file, "w")
        f.write('python "%%~dp0\%s" %%*\n' % split(script)[1])
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)

version = "2017.1.0"

url = "https://bitbucket.org/fenics-project/instant/"
tarball = None
if not 'dev' in version:
    tarball = url + "downloads/instant-%s.tar.gz" % version

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Operating System :: POSIX
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Topic :: Scientific/Engineering :: Mathematics
Topic :: Software Development :: Libraries :: Python Modules
"""

requires = ["numpy", "six"]
if sys.version_info[0] == 2:
    requires.append("subprocess32")

setup(name="instant",
      version=version,
      description="Instant Inlining of C/C++ in Python",
      author="Magne Westlie, Kent-Andre Mardal, Martin Sandve Alnes and Ilmar M. Wilbers",
      author_email="fenics-dev@googlegroups.com",
      classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
      url=url,
      download_url=tarball,
      packages=['instant'],
      package_dir={'instant': 'instant'},
      package_data={'': [join('swig', 'numpy.i')]},
      scripts=scripts,
      install_requires=requires,
      data_files=[(join("share", "man", "man1"),
                   [join("doc", "man", "man1", "instant-clean.1.gz"),
                    join("doc", "man", "man1", "instant-showcache.1.gz")])]
      )
