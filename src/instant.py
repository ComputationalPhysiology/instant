"""
By using the class instant a Python extension module can
be created at runtime. For the user, it behaves somewhat like
an inline module, except you have to import the module manually.

The code can be either C or C++, but like when programming C or C++,
it has to be inside a function or a similar C/C++ construct.

A simple example: (see test1.py)

>>> from instant import inline
>>> add_func = inline(\"double add(double a, double b){ return a+b; }\")
>>> print "The sum of 3 and 4.5 is ", add_func(3, 4.5)

"""


import os, sys,re
import commands 
import string
import md5
import shutil




VERBOSE = 1
USE_CACHE=0 
COPY_LOCAL_FILES=0

def get_instant_dir():
    instant_dir = '.'
    if USE_CACHE: 
        instant_dir = os.path.join((os.environ['HOME']), ".instant")
    return instant_dir

def get_tmp_dir(): 
    tmp_dir = '.'
    if USE_CACHE: 
        tmp_dir = os.path.join("/tmp/instant") 
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
    return tmp_dir


def path_walk_callback(arg, directory, files):
    stack = []
    tmp_dir = get_tmp_dir() 
    for file in files:
        if not directory == tmp_dir: 
            if file[-3:] == "md5":
                f = open(os.path.join(directory,file))
                line = f.readline()
                if arg[0] == line:
                    arg.append(directory)                                                                    


def find_module(md5sum):  
    list = [md5sum]
    instant_dir = get_instant_dir()
    os.path.walk(instant_dir, path_walk_callback, list)
    if len(list) == 2:                                                                     
        dir = list[1]
        sys.path.insert(0,os.path.join(instant_dir, md5sum,dir)) 
        return 1 
    return 0





class instant:
    # Default values:

    def __init__(self):
        """ instant constructor """
        self.code         = """
void f()
{
  printf("No code supplied!\\n");
}"""
        self.gen_setup  = 1 
        self.module  = 'instant_swig_module'
        self.swigopts     = '-I.'
        self.init_code    = '  //Code for initialisation here'
        self.system_headers = []
        self.local_headers  = []
        self.wrap_headers   = []
        self.sources        = []
        self.include_dirs   = ['.']
        self.libraries      = []
        self.library_dirs   = []
        self.cppargs        = '-O2'
        self.object_files   = []
        self.arrays         = []
        self.additional_definitions = ""
        self.additional_declarations = ""
        self.generate_Interface = True


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
            elif key == 'system_headers':
                self.system_headers = dict[key]
            elif key == 'local_headers':
                self.local_headers = dict[key]
                self.include_dirs.append("..")
            elif key == 'wrap_headers':
                self.wrap_headers = dict[key]
            elif key == 'include_dirs':
                self.include_dirs.extend(dict[key])
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
            elif key == 'additional_declarations':
                self.additional_declarations = dict[key]
            elif key == 'generate_Interface': 
                self.generate_Interface= dict[key]


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
              - Code that should be executed when the instant extension is imported. String.
           - B{system_headers}:
              - A list of system header files required by the instant code. 
           - B{local_headers}:
              - A list of local header files required by the instant code. 
           - B{wrap_headers}:
              - A list of local header files that should be wrapped by SWIG.
           - B{include_dirs}:
              - A list of directories to search for header files.
           - B{sources}:
              - A list of source files to compile and link with the extension.
           - B{cppargs}:
              - Flags like C{-D}, C{-U}, etc. String.
           - B{libraries}:
              - A list of libraries needed by the instant extension.
           - B{library_dirs}:
              - A list of directories to search for libraries (C{-l}).
           - B{object_files}:
              - If you want to compile the files yourself. NOT YET SUPPORTED.
           - B{arrays}:
              - A list of the C arrays to be made from NumPy arrays.
           -B{additional_definitions}:
              - A list of additional definitions (typically needed for 
                inheritance) 
           -B{additional_declarations}:
              - A list of additional declarations (typically needed for 
                inheritance) 
           
        """
        if self.parse_args(args):
            print 'Nothing done!'
            return
#        self.debug()
        module_path = self.module
        previous_path = os.getcwd()
        instant_dir = get_instant_dir()


        # create list of files that should be copyied
        files_to_copy = []
        files_to_copy.extend(self.sources) 
        files_to_copy.extend(self.local_headers)
        files_to_copy.extend(self.object_files)
        files_to_copy.extend(self.wrap_headers)

        #copy files either to cache or to local directory
        if USE_CACHE: 
            #ensure that the cache dir exists
            if not os.path.isdir(instant_dir):
                os.mkdir(instant_dir)
            #ensure that the tmp dir exists
            tmp_dir = get_tmp_dir () 
            if not os.path.isdir(tmp_dir): 
                os.mkdir(tmp_dir)
            module_path = os.path.join(tmp_dir, self.module) 
            if not os.path.isdir(module_path): 
                os.mkdir(module_path)

            for file in files_to_copy: 
                shutil.copyfile(file, os.path.join(tmp_dir, self.module,  file))
        else: 
            if not os.path.isdir(module_path): 
                os.mkdir(module_path)
            if COPY_LOCAL_FILES: 
                for file in files_to_copy: 
                    shutil.copyfile(file, os.path.join(self.module,  file))




        os.chdir(module_path)
        f = open("__init__.py", 'w')
        f.write("from %s import *"% self.module)
        
        md5sum = 0 
        if self.generate_Interface: 
            self.generate_Interfacefile()
            if self.check_md5sum():
                os.chdir(previous_path)
                return 1 
        else: 
            if os.path.isfile(self.module + ".md5"): 
                os.remove(self.module + ".md5")

        if sys.platform=='win32':
            null='nul'
        else:
            null='/dev/null'
        output_file = open("compile.log",  'w')
        (swig_stat, swig_out) = commands.getstatusoutput("swig -version")
        #if (os.system("swig -version 2> %s" % null ) == 0 ):   

        if (swig_stat == 0):   
            if ( not self.gen_setup ):   
                self.generate_Makefile()
                if os.path.isfile(self.makefile_name):
                    os.system("make -f "+self.makefile_name+" clean")
                os.system("make -f "+self.makefile_name+" >& "+self.logfile_name)
                if VERBOSE >= 9:
                    os.remove(self.logfile_name)
            else: 
                self.generate_setup()
                if VERBOSE > 0:
                    print "--- Instant: compiling ---" 
                cmd = "python " + self.module + "_setup.py build_ext" 
                if VERBOSE > 1:
                    print cmd
                ret, output = commands.getstatusoutput(cmd)
                output_file.write(output)
                if not ret == 0:  
                    os.remove("%s.md5" % self.module)
                    os.chdir(previous_path)
                    raise RuntimeError, "The extension module did not compile, check %s/compile.log" % self.module 
                else: 
#                    cmd = "python " + self.module + "_setup.py install --install-platlib=. >& compile.log 2>&1" 
                    cmd = "python " + self.module + "_setup.py install --install-platlib=." 
                    if VERBOSE > 1:
                        print cmd
                    ret, output = commands.getstatusoutput(cmd) 
                    output_file.write(output)
                    if not ret == 0:  
                        os.remove("%s.md5" % self.module)
                        os.chdir(previous_path)
                        raise RuntimeError, "Could not install the  extension module, check %s/compile.log" % self.module

#            print "Module name is \'"+self.module+"\'"
            os.chdir(previous_path)
        else: 
            os.chdir(previous_path)
            raise RuntimeError, "Could not find swig!\nYou can download swig from http://www.swig.org" 

        file = open(os.path.join(get_tmp_dir(), self.module, self.module + ".md5"))  
        md5sum = file.readline() 
        

        if USE_CACHE and md5sum:   
            # FIXME os.environ['HOME'] portable ?  
            instant_dir = get_instant_dir() 
            if not os.path.isdir(instant_dir):   
                os.mkdir(instant_dir) 
            shutil.copytree(os.path.join(get_tmp_dir(), self.module), os.path.join(instant_dir, md5sum))

            try: 
                found = find_module(md5sum)
            except Exception, e: 
                print  e
            



    def debug(self):
        """
        print out all instance variable
        """
        print 'DEBUG CODE:'
        print 'code',self.code
        print 'module',self.module
        print 'swigopts',self.swigopts
        print 'init_code',self.init_code
        print 'system_headers',self.system_headers
        print 'local_headers',self.local_headers
        print 'wrap_headers',self.wrap_headers
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
         - system_headers (A list of system headers with declarations needed)
         - local_headers (A list of local headers with declarations needed)
         - wrap_headers (A list of local headers that will be wrapped by SWIG)

        """
        if VERBOSE > 1:
            print "\nGenerating interface file \'"+ self.ifile_name +"\':"
    

        func_name = self.code[:self.code.index(')')+1]


        # create typemaps 
        typemaps = "" 
        if (len(self.arrays) > 0): 
          for a in self.arrays:  
            # 1 dimentional arrays, ie. vectors
            if (len(a) == 2):  
              typemap = """
%%typemap(in) (int %(n)s,double* %(array)s){
  if (!PyArray_Check($input)) { 
    PyErr_SetString(PyExc_TypeError, "Not a NumPy array");
    return NULL; ;
  }
  PyArrayObject* pyarray;
  pyarray = (PyArrayObject*)$input; 
  $1 = pyarray->dimensions[0];
  $2 = (double*)pyarray->data;
}
""" % { 'n' : a[0] , 'array' : a[1] }
              typemaps += typemap
            # n dimentional arrays, ie. matrices and tensors  
            elif (len(a) == 3):  
              typemap = """
%%typemap(in) (int %(n)s,int* %(ptv)s,double* %(array)s){
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
""" % { 'n' : a[0] , 'ptv' : a[1], 'array' : a[2] }
              typemaps += typemap

        self.system_headers_code = "\n".join(['#include <%s>' % header for header in self.system_headers])
        self.local_headers_code = "\n".join(['#include "%s"' % header for header in self.local_headers])
        self.wrap_headers_code1 = "\n".join(['#include "%s"' % header for header in self.wrap_headers])
        self.wrap_headers_code2 = "\n".join(['%%include "%s"' % header for header in self.wrap_headers])

        self.typemaps = typemaps 

        interface_string = """
%%module (directors="1") %(module)s

%%feature("director");

%%{
#include <iostream>
%(additional_definitions)s 
%(system_headers_code)s 
%(local_headers_code)s 
%(wrap_headers_code1)s 
%(code)s
%%}

%%feature("autodoc", "1");
%%init%%{
%(init_code)s
%%}

%(additional_definitions)s
%(additional_declarations)s
%(wrap_headers_code2)s
%(typemaps)s
%(code)s;

""" % vars(self)
     

        f = open(self.ifile_name, 'w')
        f.write(interface_string)
        f.close()
        if VERBOSE > 1:
            print '... Done'
        return func_name[func_name.rindex(' ')+1:func_name.index('(')]

    def getmd5sumfiles(self, filenames):
        '''
        get the md5 value of filename
        modified based on Python24\Tools\Scripts\md5sum.py
        '''

        m = md5.new()

        filenames.sort()


        for filename in filenames: 
         
#            print "Adding file ", filename, "to md5 sum "

            try:
                fp = open(filename, 'rb')
            except IOError, msg:
                sys.stderr.write('%s: Can\'t open: %s\n' % (filename, msg))
                return None

            try:
                while 1:
                    data = fp.read()
                    if not data:
                        break
                    m.update(data)
            except IOError, msg:
                print "filename ", filename 
                sys.stderr.write('%s: I/O error: %s\n' % (filename, msg))
                return None
            fp.close() 

        return m.hexdigest().upper()

    def writemd5sumfile(self, filenames, md5out=sys.stdout):
        result=self.getmd5sumfiles(filenames)
        try:
            fp = open(md5out, 'w')
        except IOError, msg:
            sys.stderr.write('%s: Can\'t open: %s\n' % (filename, msg))
        fp.write(result)
        fp.close()

    def check_md5sum(self): 
        """ 
        Check if the md5sum of the generated interface file has changed since the last
        time the module was compiled. If it has changed then recompilation is necessary.  
        """ 
        md5sum_files = []
        md5sum_files.append(self.ifile_name)
        for i in self.sources:       md5sum_files.append(i)
        for i in self.wrap_headers:  md5sum_files.append(i)
        for i in self.local_headers: md5sum_files.append(i)

        if (os.path.isfile(self.module+".md5")):
            current_md5sum = self.getmd5sumfiles(md5sum_files )
            if USE_CACHE and find_module(current_md5sum):
                return 1 
            else: 
                file = open(self.module + ".md5") 
                last_md5sum = file.readline()
                if current_md5sum == last_md5sum:
                    return 1 
                else: 
                    if VERBOSE > 2:  
                        print "md5sum_files ", md5sum_files
                    self.writemd5sumfile(md5sum_files, self.module + ".md5")
                    return 0 
        else:
            self.writemd5sumfile(md5sum_files, self.module + ".md5")
            current_md5sum = self.getmd5sumfiles(md5sum_files )
            if find_module(current_md5sum):
                return 1 
            else: 
                return 0
        return 0; 


    
#    def check_md5sum(self): 
#        """ 
#        Check if the md5sum of the generated interface file has changed since the last
#        time the module was compiled. If it has changed then recompilation is necessary.  
#        """ 
#        if (os.path.isfile(self.module+".md5")):
#            pipe = os.popen("md5sum " + self.ifile_name)  
#            current_md5sum = pipe.readline() 
#            file = open(self.module + ".md5") 
#            last_md5sum = file.readline()
#            if ( current_md5sum == last_md5sum) : return 1  
#            else: 
#                os.system("md5sum " + self.ifile_name +  " > " + self.module + ".md5")  
#                return 0 
#                
#            
#        else:
#            os.system("md5sum " + self.ifile_name +  " > " + self.module + ".md5")  
#            return 0
#        
#        return 0; 

    def generate_setup(self): 
        """
        Generates a setup.py file
        """
        self.cppsrcs.append( "%s_wrap.cxx" %self.module )
        f = open(self.module+'_setup.py', 'w')
        inc_dir = ""
        compile_args = ""
        if len(self.cppargs) > 1:  
            compile_args = ", extra_compile_args=['%s']" % self.cppargs 
        
        if len(self.local_headers) > 0: inc_dir = "-I.."  
        # >& compile.log
        f.write(""" 
import os
from distutils.core import setup, Extension
name = '%s' 
swig_cmd ='swig -python -c++ -O %s %s %s'
os.system(swig_cmd)
sources = %s 
setup(name = '%s', 
      ext_modules = [Extension('_' + '%s', sources, 
                     include_dirs=%s, library_dirs=%s, 
                     libraries=%s %s)])  
        """ % (self.module, inc_dir, self.swigopts, self.ifile_name, 
               self.cppsrcs, 
               self.module, self.module, self.include_dirs, self.library_dirs, self.libraries, compile_args))   
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
        
        if VERBOSE > 1:
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
        in instant.

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
              - Code that should be executed when the instant extension is imported. String.
           - B{system_headers}:
              - A list of system header files required by the instant code. 
           - B{local_headers}:
              - A list of local header files required by the instant code. 
           - B{wrap_headers}:
              - A list of local header files that will be wrapped by SWIG.
           - B{include_dirs}:
              - A list of directories to search for header files.
           - B{sources}:
              - A list of source files to compile and link with the extension.
           - B{cppargs}:
              - Flags like C{-D}, C{-U}, etc. String.
           - B{libraries}:
              - A list of libraries needed by the instant extension.
           - B{library_dirs}:
              - A list of directories to search for libraries (C{-l}).
           - B{object_files}:
              - If you want to compile the files yourself. NOT YET SUPPORTED.
           - B{arrays}:
              - A list of the C arrays to be made from NumPy arrays.
    """ 
    ext = instant()
    ext.create_extension(**args)


def inline(c_code):
    """
       This is a short wrapper around the create_extention function 
       in instant. 
       
       It creates an extension module given that
       the input is a valid C function. It is only possible
       to inline one C function each time. 

       Usage: 

       >>> from instant import inline
       >>> add_func = inline("double add(double a, double b){ return a+b; }")
       >>> print "The sum of 3 and 4.5 is ", add_func(3, 4.5)


    """
    ext = instant()
    try: 
        func = c_code[:c_code.index('(')]
        ret, func_name = func.split()
        ext.create_extension(code=c_code, module="inline_ext")
        exec("from inline_ext import %s as func_name"% func_name) 
        return func_name
    except: 
        ext.create_extension(code=c_code, module="inline_ext")
        exec("import inline_ext as I") 
        return I 



def inline_with_numpy(c_code, **args_dict):
    """
       This is a short wrapper around the create_extention function 
       in instant. 
       
       It creates an extension module given that
       the input is a valid C function. It is only possible
       to inline one C function each time. The difference between
       this function and the inline function is that C-arrays can be used. 
       The following example illustrates that. 

       Usage: 

       >>> import numpy
       >>> import time
       >>> from instant import inline_with_numpy
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

    ext = instant()
    func = c_code[:c_code.index('(')]
    ret, func_name = func.split()
    import numpy
    args_dict["code"] = c_code 
    args_dict["module"] = "inline_ext_numpy" 
    if args_dict.has_key("system_headers"):  
        args_dict["system_headers"].append ("arrayobject.h")
    else: 
        args_dict["system_headers"] = ["arrayobject.h"]

    if args_dict.has_key("include_dirs"): 
        args_dict["include_dirs"].append("%s/numpy"% numpy.get_include())
    else: 
        args_dict["include_dirs"] = ["%s/numpy"% numpy.get_include()]

    if args_dict.has_key("init_code"):
        args_dict["init_code"] += "\nimport_array();\n"
    else: 
        args_dict["init_code"] = "\nimport_array();\n"



    ext.create_extension(**args_dict)
    exec("from inline_ext_numpy import %s as func_name"% func_name) 
    return func_name


def inline_with_numeric(c_code, **args_dict):
    """
       This is a short wrapper around the create_extention function 
       in instant. 
       
       It creates an extension module given that
       the input is a valid C function. It is only possible
       to inline one C function each time. The difference between
       this function and the inline function is that C-arrays can be used. 
       The following example illustrates that. 

       Usage: 

       >>> import numpy 
       >>> import time
       >>> from instant import inline_with_numeric
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

    args_dict["code"] = c_code 
    args_dict["module"] = "inline_ext_numeric" 
    if args_dict.has_key("system_headers"):  
        args_dict["system_headers"].append ("arrayobject.h")
    else: 
        args_dict["system_headers"] = ["arrayobject.h"]

    if args_dict.has_key("include_dirs"): 
        args_dict["include_dirs"].extend( [[sys.prefix + "/include/python" + sys.version[:3] + "/Numeric", 
                                            sys.prefix + "/include" + "/Numeric"][sys.platform=='win32']])
    else: 
        args_dict["include_dirs"] = [[sys.prefix + "/include/python" + sys.version[:3] + "/Numeric", 
                                      sys.prefix + "/include" + "/Numeric"][sys.platform=='win32']]

    if args_dict.has_key("init_code"):
        args_dict["init_code"] += "\nimport_array();\n"
    else: 
        args_dict["init_code"] = "\nimport_array();\n"


    try: 
        ext = instant()
        func = c_code[:c_code.index('(')]
        ret, func_name = func.split()

        ext.create_extension(**args_dict)

        exec("from inline_ext_numeric import %s as func_name"% func_name) 
        return func_name
    except: 
        ext = instant()
        ext.create_extension(**args_dict)

        exec("import inline_ext_numeric as I") 
        return I  



def inline_with_numarray(c_code, **args_dict):
    """
       This is a short wrapper around the create_extention function 
       in instant. 
       
       It creates an extension module given that
       the input is a valid C function. It is only possible
       to inline one C function each time. The difference between
       this function and the inline function is that C-arrays can be used. 
       The following example illustrates that. 

       Usage: 

       >>> import numarray 
       >>> import time
       >>> from instant import inline_with_numarray
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

    ext = instant()
    func = c_code[:c_code.index('(')]
    ret, func_name = func.split()

    args_dict["code"] = c_code 
    args_dict["module"] = "inline_ext_numarray" 
    if args_dict.has_key("system_headers"):  
        args_dict["system_headers"].append ("arrayobject.h")
    else: 
        args_dict["system_headers"] = ["arrayobject.h"]

    if args_dict.has_key("include_dirs"): 
        args_dict["include_dirs"].extend( [[sys.prefix + "/include/python" + sys.version[:3] + "/numarray", 
                                            sys.prefix + "/include" + "/numarray"][sys.platform=='win32']])
    else: 
        args_dict["include_dirs"] = [[sys.prefix + "/include/python" + sys.version[:3] + "/numarray", 
                                      sys.prefix + "/include" + "/numarray"][sys.platform=='win32']]
    if args_dict.has_key("init_code"):
        args_dict["init_code"] += "\nimport_array();\n"
    else: 
        args_dict["init_code"] = "\nimport_array();\n"

    ext.create_extension(**args_dict)

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
        






