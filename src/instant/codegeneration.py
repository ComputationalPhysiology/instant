"""This module contains helper functions for code generation."""

import re, os
from output import instant_assert, instant_warning, instant_debug, write_file


def mapstrings(format, sequence):
    return "\n".join(format % i for i in sequence)
    

def reindent(code):
    '''Reindent a multiline string to allow easier to read syntax.
    
    Each line will be indented relative to the first non-empty line.
    Start the first line without text like shown in this example::
    
        code = reindent("""
            Foo
            Bar
                Blatti
            Ping
            """)
    
    makes all indentation relative to Foo.
    '''
    lines = code.split("\n")
    space = ""
    # Get initial spaces from first non-empty line:
    for l in lines:
        if l:
            r = re.search(r"^( [ ]*)", l)
            if r is not None:
                space = r.groups()[0]
            break
    if not space:
        return code
    n = len(space)
    instant_assert(space == " "*n, "Logic breach in reindent.")
    return "\n".join(re.sub(r"^%s" % space, "", l) for l in lines)


def write_interfacefile(filename, modulename, code, init_code,
        additional_definitions, additional_declarations,
        system_headers, local_headers, wrap_headers, arrays):
    """Generate a SWIG interface file. Intended for internal library use.
    
    The input arguments are as follows:
      - modulename (Name of the module)
      - code (Code to be wrapped)
      - init_code (Code to put in the init section of the interface file)
      - additional_definitions (Definitions to be placed in initial block with
        C code as well as in the main section of the SWIG interface file)
      - additional_declarations (Declarations to be placed in the main section
        of the SWIG interface file)
      - system_headers (A list of system headers with declarations needed by the wrapped code)
      - local_headers (A list of local headers with declarations needed by the wrapped code)
      - wrap_headers (A list of local headers that will be included in the code and wrapped by SWIG)
      - arrays (A nested list, the inner lists describing the different arrays)
    
    The result of this function is that a SWIG interface with
    the name modulename.i is written to the current directory.
    """
    instant_debug("Generating SWIG interface file '%s'." % filename)
    
    # create typemaps 
    typemaps = ""
    for a in arrays:
        if 'in' in a:
            # input arrays
            a.remove('in')
            instant_assert(len(a) > 1 and len(a) < 5, "Wrong number of elements in input array")
            if len(a) == 2:
                # 1-dimensional arrays, i.e. vectors
                typemaps += reindent("""
                %%apply (int DIM1, double* IN_ARRAY1) {(int %(n1)s, double* %(array)s)};
                """ % { 'n1' : a[0], 'array' : a[1] })
            elif len(a) == 3:
                # 2-dimensional arrays, i.e. matrices
                typemaps += reindent("""
                %%apply (int DIM1, int DIM2, double* IN_ARRAY2) {(int %(n1)s, int %(n2)s, double* %(array)s)};
                """ % { 'n1' : a[0], 'n2' : a[1], 'array' : a[2] })
            else:
                # 3-dimensional arrays, i.e. tensors
                typemaps += reindent("""
                %%apply (int DIM1, int DIM2, int DIM3, double* IN_ARRAY3) {(int %(n1)s, int %(n2)s, int %(n3)s, double* %(array)s)};
                """ % { 'n1' : a[0], 'n2' : a[1], 'n3' : a[2], 'array' : a[3] })
        elif 'out' in a:
            # output arrays
            a.remove('out')
            instant_assert(len(a) > 1 and len(a) < 3, "Output array must be 1-dimensional")
            # 1-dimensional arrays, i.e. vectors
            typemaps += reindent("""
            %%apply (int DIM1, double* ARGOUT_ARRAY1) {(int %(n1)s, double* %(array)s)};
            """ % { 'n1' : a[0], 'array' : a[1] })
        else:
            # in-place arrays
            instant_assert(len(a) > 1 and len(a) < 5, "Wrong number of elements in output array")
            if 'multi' in a:
                # n-dimensional arrays, i.e. tensors > 3-dimensional
                a.remove('multi')
                typemaps += reindent("""
                %%typemap(in) (int %(n)s,int* %(ptv)s,double* %(array)s){
                  if (!PyArray_Check($input)) { 
                    PyErr_SetString(PyExc_TypeError, "Not a NumPy array");
                    return NULL; ;
                  }
                  PyArrayObject* pyarray;
                  pyarray = (PyArrayObject*)$input; 
                  $1 = int(pyarray->nd);
                  int* dims = new int($1); 
                  for (int d=0; d<$1; d++) {
                     dims[d] = int(pyarray->dimensions[d]);
                  }
            
                  $2 = dims;  
                  $3 = (double*)pyarray->data;
                }
                %%typemap(freearg) (int %(n)s,int* %(ptv)s,double* %(array)s){
                    // deleting dims
                    delete $2; 
                }
                """ % { 'n' : a[0] , 'ptv' : a[1], 'array' : a[2] })
            elif len(a) == 2:
                # 1-dimensional arrays, i.e. vectors
                typemaps += reindent("""
                %%apply (int DIM1, double* INPLACE_ARRAY1) {(int %(n1)s, double* %(array)s)};
                """ % { 'n1' : a[0], 'array' : a[1] })
            elif len(a) == 3:
                # 2-dimensional arrays, i.e. matrices
                typemaps += reindent("""
                %%apply (int DIM1, int DIM2, double* INPLACE_ARRAY2) {(int %(n1)s, int %(n2)s, double* %(array)s)};
                """ % { 'n1' : a[0], 'n2' : a[1], 'array' : a[2] })
            else:
                # 3-dimensional arrays, i.e. tensors
                typemaps += reindent("""
                %%apply (int DIM1, int DIM2, int DIM3, double* INPLACE_ARRAY3) {(int %(n1)s, int %(n2)s, int %(n3)s, double* %(array)s)};
                """ % { 'n1' : a[0], 'n2' : a[1], 'n3' : a[2], 'array' : a[3] })
            # end
        # end if
    # end for
    
    system_headers_code = mapstrings('#include <%s>', system_headers)
    local_headers_code  = mapstrings('#include "%s"', local_headers)
    wrap_headers_code1  = mapstrings('#include "%s"', wrap_headers)
    wrap_headers_code2  = mapstrings('%%include "%s"', wrap_headers)
    
    interface_string = reindent("""
        %%module  %(modulename)s
        //%%module (directors="1") %(modulename)s

        //%%feature("director");

        %%{
        #include <iostream>
        %(additional_definitions)s 
        %(system_headers_code)s 
        %(local_headers_code)s 
        %(wrap_headers_code1)s 
        %(code)s
        %%}

        //%%feature("autodoc", "1");
        %%include "numpy.i"
        
        %%init%%{
        %(init_code)s
        %%}

        %(additional_definitions)s
        %(additional_declarations)s
        %(wrap_headers_code2)s
        //%(typemaps)s
        %(code)s;

        """ % locals())
    
    write_file(filename, interface_string)
    instant_debug("Done generating interface file.")


def write_setup(filename, modulename, csrcs, cppsrcs, local_headers, include_dirs, library_dirs, libraries, swig_include_dirs, swigargs, cppargs, lddargs):
    """Generate a setup.py file. Intended for internal library use."""
    instant_debug("Generating %s." % filename)

    # FIXME: This must be considered a hack, fix later:
    import instant
    prefix = os.path.sep.join(instant.__file__.split(os.path.sep)[:-5])
    swig_include_dirs.append(os.path.join(prefix, "include", "instant", "swig"))
    
    # Handle arguments
    swigfilename = "%s.i" % modulename
    wrapperfilename = "%s_wrap.cxx" % modulename
    
    # Treat C and C++ files in the same way for now
    cppsrcs = cppsrcs + csrcs + [wrapperfilename]
    
    swig_args = ""
    if swigargs:
        swig_args = " ".join(swigargs)

    compile_args = ""
    if cppargs:  
        compile_args = ", extra_compile_args=%r" % cppargs 

    link_args = ""
    if lddargs:  
        link_args = ", extra_link_args=%r" % lddargs 

    swig_include_dirs = " ".join("-I%s"%d for d in swig_include_dirs)
    if len(local_headers) > 0:
        swig_include_dirs += " -I.."
    
    # Generate code
    code = reindent("""
        import os
        from distutils.core import setup, Extension
        name = '%s'
        swig_cmd =r'swig -python %s %s %s'
        os.system(swig_cmd)
        sources = %s
        setup(name = '%s',
              ext_modules = [Extension('_' + '%s',
                             sources,
                             include_dirs=%s,
                             library_dirs=%s,
                             libraries=%s %s %s)])  
        """ % (modulename, swig_include_dirs, swig_args, swigfilename, cppsrcs, 
               modulename, modulename, include_dirs, library_dirs, libraries, compile_args, link_args))
    
    write_file(filename, code)
    instant_debug("Done writing setup.py file.")


def _test_write_interfacefile():
    modulename = "testmodule"
    code = "void foo() {}"
    init_code = "/* custom init code */"
    additional_definitions = "/* custom definitions */"
    additional_declarations = "/* custom declarations */"
    system_headers = ["system_header1.h", "system_header2.h"]
    local_headers = ["local_header1.h", "local_header2.h"]
    wrap_headers = ["wrap_header1.h", "wrap_header2.h"]
    arrays = [["length1", "array1"], ["dims", "lengths", "array2"]]
    
    write_interfacefile("%s.i" % modulename, modulename, code, init_code, additional_definitions, additional_declarations, system_headers, local_headers, wrap_headers, arrays)
    print "".join(open("%s.i" % modulename).readlines())


def _test_write_setup():
    modulename = "testmodule"
    csrcs = ["csrc1.c", "csrc2.c"]
    cppsrcs = ["cppsrc1.cpp", "cppsrc2.cpp"]
    local_headers = ["local_header1.h", "local_header2.h"]
    include_dirs = ["includedir1", "includedir2"]
    library_dirs = ["librarydir1", "librarydir2"]
    libraries = ["lib1", "lib2"]
    swig_include_dirs = ["swigdir1", "swigdir2"],
    swigargs = ["-Swigarg1", "-Swigarg2"]
    cppargs = ["-cpparg1", "-cpparg2"]
    lddargs = ["-Lddarg1", "-Lddarg2"]
    
    write_setup("setup.py", modulename, csrcs, cppsrcs, local_headers, include_dirs, library_dirs, libraries, swig_include_dirs, swigargs, cppargs, lddargs)
    print "".join(open("setup.py").readlines())


if __name__ == "__main__":
    _test_write_interfacefile()
    print "\n"*3
    _test_write_setup()

