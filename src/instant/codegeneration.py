
import re
from output import instant_warning, instant_debug, write_file


def mapstrings(format, sequence):
    return "\n".join(format % i for i in sequence)
    

def reindent(code):
    '''Reindent a multiline string to allow easier to read syntax.
    
    Each line will be indented relative to the first non-empty line.
    Start the first line without text like shown in this example:
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
    assert space == " "*n
    return "\n".join(re.sub(r"^%s" % space, "", l) for l in lines)


def generate_interfacefile(modulename, code, init_code,
        additional_definitions, additional_declarations,
        system_headers, local_headers, wrap_headers, arrays):
    """
    Generate a SWIG interface file.
    
    The input arguments are as follows:
    - modulename (Name of the extension module)
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
    filename = "%s.i" % modulename
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


def generate_setup(modulename, csrcs, cppsrcs, local_headers, include_dirs, library_dirs, libraries, swigargs, cppargs, lddargs):
    """Generate a setup.py file.

    The arguments are as follows:
    FIXME
    """
    instant_debug("Generating setup.py.")
    instant_warning("FIXME: Not using csrcs in generate_setup().")
    
    # Handle arguments
    swigfilename = "%s.i" % modulename
    wrapperfilename = "%s_wrap.cxx" % modulename
    
    cppsrcs = cppsrcs + [wrapperfilename]
    
    compile_args = ""
    if len(cppargs) > 0:  
        compile_args = ", extra_compile_args=%s" % cppargs 

    link_args = ""
    if len(lddargs) > 0:  
        link_args = ", extra_link_args=%s" % lddargs 

    inc_dir = ""
    if len(local_headers) > 0:
        inc_dir = "-I.."
    
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
        """ % (modulename, inc_dir, swigargs, swigfilename, cppsrcs, 
               modulename, modulename, include_dirs, library_dirs, libraries, compile_args, link_args))
    
    filename = "setup.py"
    write_file(filename, code)
    instant_debug("Done writing setup.py file.")
    return filename


def generate_makefile(modulename, csrcs, cppsrcs, local_headers, include_dirs, library_dirs, libraries, swigargs, cppargs, lddargs):
    """Generates a project dependent Makefile.
    
    This makefile includes and uses SWIG's own Makefile to 
    create an extension module of the supplied C/C++ code.
    The arguments are as follows:
    FIXME
    """
    # FIXME: What's csrcs, not used in setup.py.
    instant_warning("FIXME: Not using local_headers in generate_makefile().")
    instant_warning("FIXME: Not using lddargs in generate_makefile().")
    instant_debug("Generating makefile.")
    swigfilename = "%s.i" % modulename
    code = reindent("""
        LIBS = %s
        LDPATH = 

        FLAGS = %s

        SWIG       = swig 
        SWIGOPT    = %s
        INTERFACE  = %s
        TARGET     = %s
        INCLUDES   = 

        SWIGMAKEFILE = $(SWIGSRC)/Examples/Makefile

        python::
            $(MAKE) -f '$(SWIGMAKEFILE)' INTERFACE='$(INTERFACE)' \\
            SWIG='$(SWIG)' SWIGOPT='$(SWIGOPT)'  \\
            SRCS='%s' \\
            CPPSRCS='%s' \\
            INCLUDES='$(INCLUDES) %s' \\
            LIBS='$(LIBS) %s' \\
            CFLAGS='$(CFLAGS) $(FLAGS)' \\
            TARGET='$(TARGET)' \\
            python_cpp

        clean::
            rm -f *_wrap* _%s.so *.o $(OBJ_FILES)  *~
        """ % (" ".join(libraries),
               cppargs,
               swigargs,
               swigfilename,
               modulename,
               " ".join(csrcs),
               " ".join(cppsrcs),
               " ".join(include_dirs),
               " ".join(library_dirs),
               modulename))
    # end code
    filename = "Makefile"
    write_file(filename, code)
    instant_debug("Done generating makefile, filename is '%s'." % filename)
    return filename


def _test_generate_interfacefile():
    modulename = "testmodule"
    code = "void foo() {}"
    init_code = "/* custom init code */"
    additional_definitions = "/* custom definitions */"
    additional_declarations = "/* custom declarations */"
    system_headers = ["system_header1.h", "system_header2.h"]
    local_headers = ["local_header1.h", "local_header2.h"]
    wrap_headers = ["wrap_header1.h", "wrap_header2.h"]
    arrays = [] # FIXME: Example input here
    
    generate_interfacefile(modulename, code, init_code, additional_definitions, additional_declarations, system_headers, local_headers, wrap_headers, arrays)
    print "".join(open("%s.i" % modulename).readlines())


def _test_generate_setup():
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
    
    generate_setup(modulename, csrcs, cppsrcs, local_headers, include_dirs, library_dirs, libraries, swigargs, cppargs, lddargs)
    print "".join(open("setup.py").readlines())


def _test_generate_makefile():
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
    
    generate_makefile(modulename, csrcs, cppsrcs, local_headers, include_dirs, library_dirs, libraries, swigargs, cppargs, lddargs)
    print "".join(open("Makefile").readlines())


if __name__ == "__main__":
    _test_generate_interfacefile()
    print "\n"*3
    _test_generate_setup()
    print "\n"*3
    _test_generate_makefile()


