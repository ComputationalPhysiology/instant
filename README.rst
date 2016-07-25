=======
Instant
=======

Instant is a Python module that allows for instant inlining of C and
C++ code in Python. It is a small Python module built on top of SWIG
and Distutils. For more information, visit:

https://bitbucket.org/fenics-project/instant


Documentation
=============

Instant documentation can be viewed at
http://fenics-instant.readthedocs.org/

.. image:: https://readthedocs.org/projects/fenics-instant/badge/?version=latest
   :target: http://fenics.readthedocs.io/projects/instant/en/latest/?badge=latest
   :alt: Documentation Status


Dependencies
============

Instant depends on Python 2.7 or later, SWIG, and NumPy


Optional dependencies
=====================

To enable NFS safe file locking flufl.lock can be installed:

https://gitlab.com/warsaw/flufl.lock


Environment
===========

Instant's behaviour depened on following environment variables:

 - INSTANT_CACHE_DIR
 - INSTANT_ERROR_DIR

     These options can override placement of default cache and error
     directories in ~/.instant/cache and ~/.instant/error.

 - INSTANT_SYSTEM_CALL_METHOD

     Choose method for calling external programs (pkgconfig,
     swig, cmake, make). Available values:

       - 'SUBPROCESS'

           Uses pipes. Probably not OFED-fork safe. Default.

       - 'OS_SYSTEM'

           Uses temporary files. Probably OFED-fork safe.
