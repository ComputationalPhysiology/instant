"""
By using the class Instant a Python extension module can
be created at runtime. For the user, it behaves somewhat like
an inline module, except you have to import the module manually.

The code can be either C or C++, but like when programming C or C++,
it has to be inside a function or a similar C/C++ construct.

A simple example: (see test1.py)

>>> from Instant import inline
>>> add_func = inline(\"double add(double a, double b){ return a+b; }\")
>>> print "The sum of 3 and 4.5 is ", add_func(3, 4.5)

"""


import os, sys,re
import commands 
import string




VERBOSE = 0



class Instant:
    # Default values:
    code         = """
void f()
{
  printf("No code supplied!\\n");
}"""
    gen_setup  = 1 
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
    arrays       = []
    additional_definitions = ""

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
	    elif key == 'arrays':
                self.arrays = dict[key]
	    elif key == 'additional_definitions':
                self.additional_definitions = dict[key]


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
           - B{arrays}:
              - A list of the C arrays to be made from NumPy arrays.

           
        """
        if self.parse_args(args):
            print 'Nothing done!'
            return
#        self.debug()
        if not os.path.isdir(self.module): 
            os.mkdir(self.module)
        os.chdir(self.module)
        f = open("__init__.py", 'w')
        f.write("from %s import *"% self.module)
        
        self.generate_Interfacefile()
	if self.check_md5sum(): return 1 
	if (os.system("swig -version 2> /dev/null ") == 0 ):   
   	    if ( not self.gen_setup ):   
                self.generate_Makefile()
                if os.path.isfile(self.makefile_name):
                    os.system("make -f "+self.makefile_name+" clean")
                os.system("make -f "+self.makefile_name+" &> "+self.logfile_name)
                if VERBOSE == 9:
                    os.remove(self.logfile_name)
	    else: 
                self.generate_setup()
	        os.system("python " + self.module + "_setup.py build_ext")
	        os.system("python " + self.module + "_setup.py install --install-platlib=.")
#            print "Module name is \'"+self.module+"\'"
	else: 
	    raise RuntimeError, "Could not find swig!\nYou can download swig from http://www.swig.org" 


    def debug(self):
        """
	print out all instance variable
	"""
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
	if ( not gen_setup ) :  
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


        typemaps = "" 
	if (len(self.arrays) > 0): 
          for a in self.arrays:  
            if (len(a) == 2):  
      	      n = a[0]
	      array = a[1]

  	      typemap = """
%stypemap(in) (int %s,double* %s){
  if (!PyArray_Check($input)) { 
    PyErr_SetString(PyExc_TypeError, "Not a NumPy array");
    return NULL; ;
  }
  PyArrayObject* pyarray;
  pyarray = (PyArrayObject*)$input; 
  $1 = pyarray->dimensions[0];
  $2 = (double*)pyarray->data;
}
""" % ('%',n,array)
              typemaps += typemap
            elif (len(a) == 3):  
      	      n = a[0]
	      ptv = a[1]
	      array = a[2]

  	      typemap = """
%stypemap(in) (int %s,int* %s,double* %s){
  if (!PyArray_Check($input)) { 
    PyErr_SetString(PyExc_TypeError, "Not a NumPy array");
    return NULL; ;
  }
  PyArrayObject* pyarray;
  pyarray = (PyArrayObject*)$input; 
  $1 = pyarray->nd;
  $2 = pyarray->dimensions;
  $3 = (double*)pyarray->data;
}
""" % ('%',n,ptv,array)
              typemaps += typemap



    
        f = open(self.ifile_name, 'w')
        f.write("""
%%module (directors="1") %s

%%feature("director");

%%{
""" % self.module)
        for header in self.headers:
            f.write("#include <%s>\n" % header)
        f.write("""
#include <iostream>
%s

%%}

%%feature("autodoc", "1");

%%init%%{
%s
%%}

%s
%s
%s;
    """ % (self.code, self.init_code, self.additional_definitions, typemaps, self.code))
        f.close()
        if VERBOSE > 0:
            print '... Done'
        return func_name[func_name.rindex(' ')+1:func_name.index('(')]
    
    def check_md5sum(self): 
        """ 
        Check if the md5sum of the generated interface file has changed since the last
        time the module was compiled. If it has changed then recompilation is necessary.  
        """ 
        if (os.path.isfile(self.module+".md5")):
            pipe = os.popen("md5sum " + self.ifile_name)  
            current_md5sum = pipe.readline() 
            file = open(self.module + ".md5") 
            last_md5sum = file.readline()
            if ( current_md5sum == last_md5sum) : return 1  
            else: 
                os.system("md5sum " + self.ifile_name +  " > " + self.module + ".md5")  
                return 0 
                
            
        else:
            os.system("md5sum " + self.ifile_name +  " > " + self.module + ".md5")  
            return 0
        
        return 0; 

    def generate_setup(self): 
        """
	Generates a setup.py file
	"""
        self.cppsrcs.append( "%s_wrap.cxx" %self.module )
	f = open(self.module+'_setup.py', 'w')
	f.write(""" 
import os
from distutils.core import setup, Extension
name = '%s' 
swig_cmd ='swig -python -c++ %s %s'
os.system(swig_cmd)
sources = %s 
setup(name = '%s', 
      ext_modules = [Extension('_' + '%s', sources, 
                     include_dirs=%s, 
                     library_dirs=%s, libraries=%s)])  
	""" % (self.module, self.swigopts, self.ifile_name, 
	       self.cppsrcs, 
	       self.module, self.module, self.include_dirs, self.library_dirs, self.libraries ))   
	f.close()


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



def create_extension(**args):
    """
        This is a small wrapper around the create_extension function
        in Instant.

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
           - B{arrays}:
              - A list of the C arrays to be made from NumPy arrays.
    """ 
    ext = Instant()
    ext.create_extension(**args)


def inline(c_code):
    """
       This is a short wrapper around the create_extention function 
       in Instant. 
       
       It creates an extension module given that
       the input is a valid C function. It is only possible
       to inline one C function each time. 

       Usage: 

       >>> from Instant import inline
       >>> add_func = inline("double add(double a, double b){ return a+b; }")
       >>> print "The sum of 3 and 4.5 is ", add_func(3, 4.5)


    """
    ext = Instant()
    func = c_code[:c_code.index('(')]
    ret, func_name = func.split()
    ext.create_extension(code=c_code, module="inline_ext")
    exec("from inline_ext import %s as func_name"% func_name) 
    return func_name


def inline_with_numpy(c_code, **args_dict):
    """
       This is a short wrapper around the create_extention function 
       in Instant. 
       
       It creates an extension module given that
       the input is a valid C function. It is only possible
       to inline one C function each time. The difference between
       this function and the inline function is that C-arrays can be used. 
       The following example illustrates that. 

       Usage: 

       >>> import numpy
       >>> import time
       >>> from Instant import inline_with_numeric
       >>> c_code = \"\"\"
           double sum (int n1, double* array1){
               double tmp = 0.0; 
               for (int i=0; i<n1; i++) {  
                   tmp += array1[i]; 
               }
               return tmp; 
           }
           \"\"\"
       >>> sum_func = inline_with_numpy(c_code,  arrays = [['n1', 'array1']])
       >>> a = numpy.arange(10000000); a = numpy.sin(a)
       >>> sum_func(a)
    """

    ext = Instant()
    func = c_code[:c_code.index('(')]
    ret, func_name = func.split()
    import numpy	
    ext.create_extension(code=c_code, module="inline_ext_numpy", 
                         headers=["arrayobject.h"], cppargs='-O3',
                         include_dirs= ["%s/numpy"% numpy.get_include()],
                         init_code='import_array();', arrays = args_dict["arrays"])
    exec("from inline_ext_numpy import %s as func_name"% func_name) 
    return func_name


def inline_with_numeric(c_code, **args_dict):
    """
       This is a short wrapper around the create_extention function 
       in Instant. 
       
       It creates an extension module given that
       the input is a valid C function. It is only possible
       to inline one C function each time. The difference between
       this function and the inline function is that C-arrays can be used. 
       The following example illustrates that. 

       Usage: 

       >>> import numpy 
       >>> import time
       >>> from Instant import inline_with_numeric
       >>> c_code = \"\"\"
           double sum (int n1, double* array1){
               double tmp = 0.0; 
               for (int i=0; i<n1; i++) {  
                   tmp += array1[i]; 
               }
               return tmp; 
           }
           \"\"\"
       >>> sum_func = inline_with_numeric(c_code,  arrays = [['n1', 'array1']])
       >>> a = numpy.arange(10000000); a = numpy.sin(a)
       >>> sum_func(a)
    """

    ext = Instant()
    func = c_code[:c_code.index('(')]
    ret, func_name = func.split()
    ext.create_extension(code=c_code, module="inline_ext_numeric", 
                         headers=["arrayobject.h"], cppargs='-O3',
                         include_dirs= [sys.prefix + "/include/python" 
                                     + sys.version[:3] + "/Numeric"],
                         init_code='import_array();', arrays = args_dict["arrays"])
    exec("from inline_ext_numeric import %s as func_name"% func_name) 
    return func_name


def inline_with_numarray(c_code, **args_dict):
    """
       This is a short wrapper around the create_extention function 
       in Instant. 
       
       It creates an extension module given that
       the input is a valid C function. It is only possible
       to inline one C function each time. The difference between
       this function and the inline function is that C-arrays can be used. 
       The following example illustrates that. 

       Usage: 

       >>> import numarray 
       >>> import time
       >>> from Instant import inline_with_numarray
       >>> c_code = \"\"\"
           double sum (int n1, double* array1){
               double tmp = 0.0; 
               for (int i=0; i<n1; i++) {  
                   tmp += array1[i]; 
               }
               return tmp; 
           }
           \"\"\"
       >>> sum_func = inline_with_numarray(c_code,  arrays = [['n1', 'array1']])
       >>> a = numarray.arange(10000000); a = numarray.sin(a)
       >>> sum_func(a)
    """

    ext = Instant()
    func = c_code[:c_code.index('(')]
    ret, func_name = func.split()
    ext.create_extension(code=c_code, module="inline_ext_numarray", 
                         headers=["arrayobject.h"], cppargs='-O3',
                         include_dirs= [sys.prefix + "/include/python" 
                                     + sys.version[:3] + "/numarray"],
                         init_code='import_array();', arrays = args_dict["arrays"])
    exec("from inline_ext_numarray import %s as func_name"% func_name) 
    return func_name






def header_and_libs_from_pkgconfig(*packages):
    """
    This function returns list of include files, flags, libraries and library directories obtain from a pkgconfig file. 
    The usage is: 
    (includes, flags, libraries, libdirs) = header_and_libs_from_pkgconfig(list_of_packages)

    """
    includes = []
    flags = []
    libs = []
    libdirs = []
    for pack in packages:
#        print commands.getstatusoutput("pkg-config --exists %s " % pack)
        if  commands.getstatusoutput("pkg-config --exists %s " % pack)[0] == 0: 
            tmp = string.split(commands.getoutput("pkg-config --cflags-only-I %s " % pack ))  
            for i in tmp: includes.append(i[2:]) 
            tmp = string.split(commands.getoutput("pkg-config --cflags-only-other %s " % pack ))  
            for i in tmp: flags.append(i) 
            tmp = string.split(commands.getoutput("pkg-config --libs-only-l  %s " % pack ))  
            for i in tmp: libs.append(i[2:]) 
            tmp = string.split(commands.getoutput("pkg-config --libs-only-L  %s " % pack ))  
            for i in tmp: libdirs.append(i[2:]) 
        else: 
            raise OSError, "The pkg-config file %s does not exist" % pack  


    return (includes,flags,libs, libdirs) 
        






