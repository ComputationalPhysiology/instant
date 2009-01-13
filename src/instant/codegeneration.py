"""This module contains helper functions for code generation."""

import re
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
      - additional_definitions (FIXME: comment)
      - additional_declarations (FIXME: comment)
      - system_headers (A list of system headers with declarations needed by the wrapped code)
      - local_headers (A list of local headers with declarations needed by the wrapped code)
      - wrap_headers (A list of local headers that will be included in the code and wrapped by SWIG)
      - arrays (FIXME: comment)
    
    The result of this function is that a SWIG interface with
    the name modulename.i is written to the current directory.
    """
    instant_debug("Generating SWIG interface file '%s'." % filename)
    
    # create typemaps 
    typemaps = ""
    for a in arrays:  
        # 1 dimentional arrays, ie. vectors
        if (len(a) == 2):  
            typemaps += reindent("""
                %%typemap(in) (int %(n)s,double* %(array)s){
                  if (!PyArray_Check($input)) { 
                    PyErr_SetString(PyExc_TypeError, "Not a NumPy array");
                    return NULL; ;
                  }
                  PyArrayObject* pyarray;
                  pyarray = (PyArrayObject*)$input; 
                  $1 = int(pyarray->dimensions[0]);
                  $2 = (double*)pyarray->data;
                }
                """ % { 'n' : a[0] , 'array' : a[1] })
        # n dimentional arrays, ie. matrices and tensors  
        elif (len(a) == 3):  
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
    #instant_warning("FIXME: Not using csrcs in write_setupfile().")
    
    # Handle arguments
    swigfilename = "%s.i" % modulename
    wrapperfilename = "%s_wrap.cxx" % modulename
    
    cppsrcs = cppsrcs + [wrapperfilename]
    
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
        swig_cmd ='swig -python %s %s %s'
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
    arrays = [] # FIXME: Example input here
    
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
    swigargs = ["-Swigarg1", "-Swigarg2"]
    cppargs = ["-cpparg1", "-cpparg2"]
    lddargs = ["-Lddarg1", "-Lddarg2"]
    
    write_setup("setup.py", modulename, csrcs, cppsrcs, local_headers, include_dirs, library_dirs, libraries, swigargs, cppargs, lddargs)
    print "".join(open("setup.py").readlines())


if __name__ == "__main__":
    _test_write_interfacefile()
    print "\n"*3
    _test_write_setup()

