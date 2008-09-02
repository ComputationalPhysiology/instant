"""
Instant allows compiled C/C++ extension modules to be created
at runtime in your Python application, using SWIG to wrap the
C/C++ code.

A simple example:
>>> from instant import inline
>>> add_func = inline(\"double add(double a, double b){ return a+b; }\")
>>> print "The sum of 3 and 4.5 is ", add_func(3, 4.5)

For more examples, see the tests/ directory in the Instant distribution.
"""

# FIXME: Metadata here and in other files.

# FIXME: Import only the official interface
from output import *
from config import *
from paths import *
from signatures import *
from cache import *
from codegeneration import *
from create_extension import *
from highlevel import *

