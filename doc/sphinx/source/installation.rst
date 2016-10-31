.. title:: Installation


============
Installation
============

Instant is normally installed as part of an installation of FEniCS.
If you are using Instant as part of the FEniCS software suite, it
is recommended that you follow the
`installation instructions for FEniCS
<https://fenics.readthedocs.io/en/latest/>`__.

To install Instant itself, read on below for a list of requirements
and installation instructions.


Requirements and dependencies
=============================

Instant requires Python version 2.7 or later and depends on the
following Python packages:

* NumPy
* SWIG

These packages will be automatically installed as part of the
installation of Instant, if not already present on your system.

If running on a cluster with Infiniband with python 2,
you also need to install a backport of the subprocess
module from python 3 to get safe fork behaviour:

* subprocess32

In addition, Instant optionally depends on flufl.lock for NFS safe
file locking flufl.lock can be installed

* flufl.lock (https://gitlab.com/warsaw/flufl.lock)


Installation instructions
=========================

To install Instant, download the source code from the
`Instant Bitbucket repository
<https://bitbucket.org/fenics-project/instant>`__,
and run the following command:

.. code-block:: console

    pip install .

To install to a specific location, add the ``--prefix`` flag
to the installation command:

.. code-block:: console

    pip install --prefix=<some directory> .


Environment
===========

Instant's behaviour depened on following environment variables:

 - ``INSTANT_CACHE_DIR``
 - ``INSTANT_ERROR_DIR``

   These options can override placement of default cache and error
   directories. The default directories are placed below the prefix
   of the currently active virtualenv or conda environment, in
   ``.cache/instant/cache`` and ``.cache/instant/error``.
   If no such environment is active, the default directories are
   ``~/.cache/instant/pythonM.N/cache`` and ``.cache/instant/pythonM.N/error``. 

 - ``INSTANT_SYSTEM_CALL_METHOD``

     Choose method for calling external programs (pkgconfig,
     swig, cmake, make). Available values:

       - ``SUBPROCESS``

           Uses pipes. Not OFED-fork safe on Python 2 unless
           subprocess32 has been installed. Default.

       - ``OS_SYSTEM``

           Uses temporary files. Probably OFED-fork safe.

.. warning:: OFED-fork safe system call method might be required to
             avoid crashes on OFED-based (InfiniBand) clusters!
             If using python 2, installing subprocess32 is recommended.
