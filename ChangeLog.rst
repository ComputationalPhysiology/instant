Changelog
=========

2017.2.0 (unreleased)
---------------------

- Nothing changed yet


2017.1.0.post1 (2017-09-12)
---------------------

- Change PyPI package name to fenics-instant. 


2017.1.0 (2017-05-09)
---------------------

- Minor fixes

2016.2.0 (2016-11-30)
---------------------

- Add Python version string to hash signature input
- Add pipelines testing, with py2 and py3 coverage
- Remove commands module (removed from Py3)
- Switch unit tests to pytest

2016.1.0 (2016-06-23)
---------------------

- Minor fixes

1.6.0 (2015-07-28)
------------------

- Minor fixes

1.5.0 (2015-01-12)
------------------

- Require Python 2.7
- Python 3 support

1.4.0 (2014-06-02)
------------------

- Introduce env var ``INSTANT_SYSTEM_CALL_METHOD`` for switching method
  for calling external programs and introduce possibly OFED-fork safe
  implementation

1.3.0 (2014-01-07)
------------------

- Add ``file_lock`` which can be used within a ``with`` statement
- Introduce ``set_log_level`` and deprecate ``set_logging_level``

1.2.0 (2013-03-24)
------------------

- Allow to use CMake instead of distutils/pkg-config

1.1.0 (2013-01-07)
------------------

- Converting python2 syntax to python3 (run 2to3)
- Patch to make instant work with python2
- Cache dir is now created if not existing

1.0.0 (2011-12-07)
------------------

- Copy all files to ``~/.instant/error/module_name`` when Instant fails
- If environment variable ``INSTANT_DISPLAY_COMPILE_LOG`` is set the
  content of ``compile.log`` will be displayed
- Removed copying of ``compile.log`` to ``~/.instant/error/``

1.0-beta2 (2011-10-28)
----------------------

- Added support for flufl.lock for NFS safe file locking

1.0-beta (2011-08-11)
---------------------

- Error log is now copied to
  ``~/.instant/error/module_name/compile.log`` and
  ``~/.instant/error/compile.log`` for easier retrieval

0.9.10 (2011-05-16)
-------------------

- Added support for setting swig binary and swig path

0.9.9 (2011-02-23)
------------------

- Optimizations
- Added support for VTK and VMTK

0.9.8 (2010-02-15)
------------------

- Fix cache related memory leak

0.9.7
-----

- Use typemaps from the NumPy SWIG interface file (numpy.i)
  enabling the use of many new data types.
- Removed support for Numeric and numarray.

0.9.6
-----

- Minor update with some new utility functions required by FFC.

0.9.5
-----

- Restructured and rewritten much of the code.
- Improved multilevel cache functionality.
- Added instant-clean and instant-showcache scripts.

0.9.4
-----

- Various new examples with swiginac and sympy implemented.
- Bug fix on 64bit. Removed director flag by default.

0.9.3
-----

- Implemented caching

0.9.2
-----

- Bug fix for the JIT in FFC

0.9.1
-----

- Added test example which demonstrate use of external C code.
- Added flag to turn of regeneration of the interface file (useful
  during debugging)

0.9
---

- Port to Windows with mingw by laserjungle, some updates by Martin
  Alnæs, and some cleanup.

0.8
---

- Added support for NumPy and Numarray.

0.7
---

- Added functionality for the use of pkg-config files.

0.6
---

- Created a more user-friendly interface

0.5
---

- Added SWIG directors for cross language inheritance

0.4
---

- Added md5sum to avoid unnecessary compilation

0.3
---

- Support for NumPy arrays

0.2
---

- Fixed bug in setup script

0.1
---

- Initial release of Instant
