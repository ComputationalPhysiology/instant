"""
Instant allows compiled C/C++ modules to be created
at runtime in your Python application, using SWIG to wrap the
C/C++ code.

A simple example:
    >>> from instant import inline
    >>> add_func = inline(\"double add(double a, double b){ return a+b; }\")
    >>> print "The sum of 3 and 4.5 is ", add_func(3, 4.5)

The main functions are C{build_module}, C{write_code}, and
C{inline*} see their documentation for more details.

For more examples, see the tests/ directory in the Instant distribution.

Questions, bugs and patches should be sent to fenics@fenicsproject.org.
"""

__authors__ = "Magne Westlie, Kent-Andre Mardal <kent-and@simula.no>, Martin Alnes <martinal@simula.no>, Ilmar M. Wilbers <ilmarw@simula.no>"
__date__ = "2014-01-07"
__version__ = "1.3.0"

# TODO: Import only the official interface
from .output import *
from .config import *
from .paths import *
from .signatures import *
from .cache import *
from .codegeneration import *
from .build import *
from .inlining import *
