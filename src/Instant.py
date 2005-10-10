"""
By using class I{Instant} from the I{instant} module, a
Python extension module can
be created at runtime. For the user, it behaves somewhat like
an inline module, except you have to import the module manually.

The code can be either C or C++, but like when programming C or C++,
it has to be inside a function or a similar C/C++ construct.

A simple example: (see test1.py)

>>> import Instant,os  
>>> ext = Instant.Instant()
>>> c_code = \"\"\"
int hello(){
  printf("Hello World!\\n");
  return 2222;
}
\"\"\"
>>> ext.create_extension(code=c_code,
                     module='test1_ext')
>>> from test1_ext import hello
>>> return_val = hello()
Hello World!
>>> print return_val 
2222



   

Another example (see test2.py) :

>>> import Instant 
>>> import Numeric
>>> a = Numeric.arange(1000000)
>>> a = Numeric.sin(a)
>>> b = Numeric.arange(1000000)
>>> b = Numeric.cos(b)
>>> s = 
\"\"\"
PyObject* add(PyObject* a_, PyObject* b_){
  /*
  various checks
  */ 
  PyArrayObject* a=(PyArrayObject*) a_;
  PyArrayObject* b=(PyArrayObject*) b_;

  int n = a->dimensions[0];

  int dims[1];
  dims[0] = n; 
  PyArrayObject* ret;
  ret = (PyArrayObject*) PyArray_FromDims(1, dims, PyArray_DOUBLE); 

  int i;
  double aj;
  double bj;
  double *retj; 
  for (i=0; i < n; i++) {
    retj = (double*)(ret->data+ret->strides[0]*i); 
    aj = *(double *)(a->data+ a->strides[0]*i);
    bj = *(double *)(b->data+ b->strides[0]*i);
    *retj = aj + bj; 
  }
return PyArray_Return(ret);
}
\"\"\"
>>> ext = Instant.Instant() 
>>> ext.create_extension(code=s, headers=["arrayobject.h"],
              include_dirs=["-I/usr/include/python2.4/Numeric"],
              init_code='import_array();', module="test2_ext"
              )
>>> import time
>>> import test2_ext 
>>> t1 = time.time() 
>>> d = test2_ext.add(a,b)
>>> t2 = time.time()
>>> print 'With Instant:',t2-t1,'seconds'
With Instant: 0.0758290290833 seconds
>>> t1 = time.time() 
>>> c = a+b
>>> t2 = time.time()
>>> print 'Med numpy: ',t2-t1,'seconds'
Med numpy:  0.0843679904938 seconds
>>> difference = c - d 
>>> sum = reduce( lambda a,b: a+b, difference)  
>>> print sum 
0.0
"""


import os, sys


VERBOSE = 0



class Instant:
    # Default values:
    code         = """
void f()
{
  printf("No code supplied!\\n");
}"""
    module  = 'instant_swig_module'
    swigopts     = '-I.'
    init_code    = '  //Code for initialisation here'
    headers      = []
    sources      = []
    include_dirs = ['-I.']
    libraries    = []
    library_dirs = []
    cppargs      = ''
    object_files = []

    def __init__(self):
        """ For now, empty! """
        pass

    def parse_args(self, dict):
        """ A method for parsing arguments. """
        for key in dict.keys():
            if key == 'code':
                self.code = dict[key]
            elif key == 'module':
                self.module = dict[key]
            elif key == 'swigopts':
                self.swigopts = dict[key]
            elif key == 'init_code':
                self.init_code = dict[key]
            elif key == 'sources':
                self.sources = dict[key]
            elif key == 'headers':
                self.headers = dict[key]
            elif key == 'include_dirs':
                self.include_dirs = dict[key]
            elif key == 'libraries':
                self.libraries = dict[key]
            elif key == 'library_dirs':
                self.library_dirs = dict[key]
            elif key == 'cppargs':
                self.cppargs = dict[key]
            elif key == 'object_files':
                self.object_files = dict[key]

        self.makefile_name = self.module+".mak"
        self.logfile_name  = self.module+".log"
        self.ifile_name    = self.module+".i"
        # sources has to be put in different variables,
        # depending on the file-suffix (.c or .cpp).
        self.srcs = []
        self.cppsrcs = []
        for file in self.sources:
            if file.endswith('.cpp'):
                self.cppsrcs.append(file)
            elif file.endswith('.c'):
                self.srcs.append(file)
            else:
                print 'Source files must have \'.c\' or \'.cpp\' suffix!'
                print 'This is not the case for',file
                return 1 # Parsing of argurments detected errors
        return 0 # all ok


    def create_extension(self, **args):
        """
        Call this function to instantly create an extension module.
        SWIG is used to generate code that can be compiled and used as
        an ordinary Python module.

        Arguments:
        ==========
           - B{code}:
              - A Python string containing C or C++ function, class, ....
           - B{module}:
              - The name you want for the module (Default is 'instant_swig_module'.). String.
           - B{swigopts}:
              - Options to swig, for instance C{-lpointers.i} to include the
                SWIG pointers.i library. String.
           - B{init_code}:
              - Code that should be executed when the Instant extension is imported. String.
           - B{headers}:
              - A list of header files required by the Instant code. 
           - B{include_dirs}:
              - A list of directories to search for header files.
           - B{sources}:
              - A list of source files to compile and link with the extension.
           - B{cppargs}:
              - Flags like C{-D}, C{-U}, etc. String.
           - B{libraries}:
              - A list of libraries needed by the Instant extension.
           - B{library_dirs}:
              - A list of directories to search for libraries (C{-l}).
           - B{object_files}:
              - If you want to compile the files yourself. NOT YET SUPPORTED.
           
        """
        if self.parse_args(args):
            print 'Nothing done!'
            return
#        self.debug()
        self.generate_Interfacefile()
        self.generate_Makefile()
        if os.path.isfile(self.makefile_name):
            os.system("make -f "+self.makefile_name+" clean")
        os.system("make -f "+self.makefile_name+" &> "+self.logfile_name)
        if VERBOSE == 9:
            os.remove(self.logfile_name)
        print "Module name is \'"+self.module+"\'"


    def debug(self):
        print 'DEBUG CODE:'
        print 'code',self.code
        print 'module',self.module
        print 'swigopts',self.swigopts
        print 'init_code',self.init_code
        print 'headers',self.headers
        print 'include_dirs',self.include_dirs
        print 'sources',self.sources
        print 'srcs',self.srcs
        print 'cppsrcs',self.cppsrcs
        print 'cppargs',self.cppargs


    def clean(self):
        """ Clean up files the current session. """
        for file in [self.module+".log",
                     self.module+".log",
                     self.module+".i",
                     self.module+".mak",
                     self.module+".py",
                     self.module+".pyc",
                     "_"+self.module+".so"]:
            if os.path.isfile(file):
                os.remove(file)

    def generate_Interfacefile(self):
        """
        Use this function to generate a SWIG interface file.
        
        To generate an interface file it uses the following class-variables:

         - code
         - ifile_name (The SWIG input file)
         - init_code (Code to put in the init section of the interface file)
         - headers (A list of headers with declarations needed)

        """
        if VERBOSE > 0:
            print "\nGenerating interface file \'"+ self.ifile_name +"\':"
    
        func_name = self.code[:self.code.index(')')+1]
    
        f = open(self.ifile_name, 'w')
        f.write("""
%%module %s

%%{
""" % self.module)
        for header in self.headers:
            f.write("   #include <%s>\n" % header)
        f.write("""
#include <iostream>

%s

%%}

%%init%%{
%s
%%}


%s;
    """ % (self.code, self.init_code, func_name))
        f.close()
        if VERBOSE > 0:
            print '... Done'
        return func_name[func_name.rindex(' ')+1:func_name.index('(')]


    def generate_Makefile(self):
        """
        Generates a project dependent Makefile, which includes and
        uses SWIG's own Makefile to create an extension module of
        the supplied C/C++ code.
        """
        f = open(self.makefile_name, 'w')
        f.write("""
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
    """ % (list2str(self.libraries),
           self.cppargs,
           self.swigopts,
           self.ifile_name,
           self.module,
           list2str(self.srcs),
           list2str(self.cppsrcs),
           list2str(self.include_dirs),
           list2str(self.library_dirs),
           self.module
           ))
        f.close()
        
        if VERBOSE > 0:
            print 'Makefile', self.makefile_name, 'generated'
            

# convert list values to string
def list2str(list):
    s = str(list)
    for c in ['[', ']', ',', '\'']:
        s = s.replace(c, '')
    return s
