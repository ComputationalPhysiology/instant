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


import os, sys, re
import commands 
import string
import md5
import shutil
import tempfile


VERBOSE = 1
#USE_CACHE=0 
COPY_LOCAL_FILES=1


def get_instant_dir():
    # os.path.expanduser works for Windows, Linux, and Mac
    # In Windows, $HOME is os.environ['HOMEDRIVE'] + os.environ['HOMEPATH']
    instant_dir = os.path.join(os.path.expanduser('~'), ".instant")
    if not os.path.isdir(instant_dir):
        os.mkdir(instant_dir)
    return instant_dir


def get_tmp_dir(): 
    tmp_dir = os.path.join(tempfile.gettempdir(), "instant") 
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    return tmp_dir


def get_instant_module_dir(md5sum):
    return os.path.join(get_instant_dir(), "instant_module_" + md5sum)


def get_md5sum_from_signature(signature):
    '''
    get the md5 value of signature 
    '''
    m = md5.new()
    m.update(signature)
    return m.hexdigest().upper()


def get_md5sum_from_files(filenames):
    '''
    get the md5 value of filename
    modified based on Python24\Tools\Scripts\md5sum.py
    '''
    m = md5.new()
    for filename in sorted(filenames): 
        #print "Adding file ", filename, "to md5 sum "
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


def write_md5sum_file(sum, md5out=sys.stdout):
    try:
        fp = open(md5out, 'w')
    except IOError, msg:
        sys.stderr.write('%s: Can\'t open: %s\n' % (filename, msg))
    fp.write(sum)
    fp.close()


def path_walk_callback(arg, directory, files):
    if directory != get_tmp_dir():
        # find .md5 file and compare its contents with given md5 sum in arg
        md5sum = arg[0]
        for filename in files:
            if filename[-3:] == "md5":
                f = open(os.path.join(directory, filename))
                line = f.readline()
                if md5sum == line:
                    arg.append(directory)                                                                    


def find_module(md5sum):
    arg = [md5sum]
    instant_dir = get_instant_dir()
    os.path.walk(instant_dir, path_walk_callback, arg)
    if len(arg) == 2:
        assert arg[1]
        directory = os.path.join(get_instant_module_dir(md5sum), arg[1])
        if VERBOSE > 9: print "find_module: directory = ", directory
        # add found module directory to path
        if not directory in sys.path:
            if VERBOSE > 9: print "Inserting directory in sys.path: ", directory
            sys.path.insert(0, directory) 
        # return module (directory) name
        if VERBOSE > 9: print "find_module returning:", os.path.split(arg[1])[-1]
        return os.path.split(arg[1])[-1]
        #return 1
    return None


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
        self.swigopts     = ' -c++ -fcompact -O -I. -small'
        self.init_code    = '  //Code for initialisation here'
        self.system_headers = []
        self.local_headers  = []
        self.wrap_headers   = []
        self.sources        = []
        self.include_dirs   = ['.']
        self.libraries      = []
        self.library_dirs   = []
        self.cppargs        = ['-O2']
        self.lddargs        = []
        self.object_files   = []
        self.arrays         = []
        self.additional_definitions = ""
        self.additional_declarations = ""
        self.generate_Interface = True
        self.signature          = ""
        self.use_cache          = False

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
                assert isinstance(dict[key], (str,tuple, list)), "Wrong type of argument to cppargs" 
                if isinstance(dict[key], str):
                    if dict[key] is "":
                        self.cppargs = []
                    else:
                        self.cppargs = [dict[key].strip()]
                elif isinstance(dict[key], (tuple, list)):
                    self.cppargs = [s.strip() for s in dict[key]] 
            elif key == 'lddargs':
                assert isinstance(dict[key], (str,tuple, list)), "Wrong type of argument to lddargs" 
                if isinstance(dict[key], str): 
                    if dict[key] is "":
                        self.lddargs = []
                    else:
                        self.lddargs = [dict[key].strip()]
                elif isinstance(dict[key], (tuple, list)):
                    self.lddargs = [s.strip() for s in dict[key]] 
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
            elif key == 'signature': 
                self.signature = dict[key]
            elif key == 'use_cache':
                self.use_cache = dict[key]

        if self.use_cache:
            self.instant_dir = get_instant_dir()
            self.get_tmp_dir = get_tmp_dir()
        else:
            self.instant_dir=''
            self.get_tmp_dir=''

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
              - Options to swig, for instance C{-lpointers.i} to include the SWIG pointers.i library. String.
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
           - B{lddargs}:
              - Flags like C{-D}, C{-U}, etc. String.
           - B{libraries}:
              - A list of libraries needed by the instant extension.
           - B{library_dirs}:
              - A list of directories to search for libraries (C{-l}).
           - B{object_files}:
              - If you want to compile the files yourself. NOT YET SUPPORTED.
           - B{arrays}:
              - A list of the C arrays to be made from NumPy arrays.
           - B{additional_definitions}:
              - A list of additional definitions (typically needed for inheritance) 
           - B{additional_declarations}:
              - A list of additional declarations (typically needed for inheritance) 
           - B{generate_Interface}:
              - Indicate if you want to generate the interface files. Bool.
           - B{signature}:
              - A signature string to identify the form instead of the source code.
           - B{use_cache}:
              - Indicate if you want to store the generated module for later use. Bool.
        """
        if self.parse_args(args):
            print 'Nothing done!' # Martin: What does this mean?
            return
        #self.debug()
        
        previous_path = os.getcwd()
        
        # create module path, either in cache or a local directory
        module_path = os.path.join(self.get_tmp_dir, self.module) 
        if not os.path.isdir(module_path): 
            os.mkdir(module_path)
        
        # copy files either to cache or to local directory
        if self.use_cache:
            # create list of files that should be copied
            files_to_copy = []
            files_to_copy.extend(self.sources) 
            files_to_copy.extend(self.local_headers)
            files_to_copy.extend(self.object_files)
            files_to_copy.extend(self.wrap_headers)
            
            # hack to keep existing behaviour:
            # (it might be a good idea to clean up the 'user interface' and behaviour
            # specifications at some point before instant 1.0 is released!)
            if VERBOSE > 9: print "Copying files: ", files_to_copy, " to ", module_path
            if self.use_cache:
                for file in files_to_copy:
                    shutil.copyfile(file, os.path.join(module_path, file))
            else:
                for file in files_to_copy:
                    shutil.copyfile(os.path.join(self.module, file), os.path.join(module_path, file))
        
        # generate __init__.py which imports compiled module contents
        os.chdir(module_path)
        f = open("__init__.py", 'w')
        f.write("from %s import *"% self.module)
        f.close()
        
        # generate interface files if wanted
        if self.generate_Interface:
            self.generate_Interfacefile()
            if self.check_md5sum():
                os.chdir(previous_path)
                return 1 # Martin: What does return 1 mean?
        else:
            # Martin: If we don't generate the interface,
            #         we remove the .md5 file, what is the logic in that?
            if os.path.isfile(self.module + ".md5"):
                os.remove(self.module + ".md5")
        
        #if sys.platform=='win32':
        #    null = 'nul'
        #else:
        #    null = '/dev/null'
        #if os.system("swig -version 2> %s" % null) == 0:
        
        try:
            # the next steps require swig!
            (swig_stat, swig_out) = commands.getstatusoutput("swig -version")
            if swig_stat != 0:
                raise RuntimeError("Could not find swig! You can download swig from http://www.swig.org")
            
            # generate Makefile or setup.py and run it
            if not self.gen_setup:
                self.generate_Makefile()
                if os.path.isfile(self.makefile_name):
                    os.system("make -f "+self.makefile_name+" clean")
                os.system("make -f "+self.makefile_name+" >& "+self.logfile_name)
                if VERBOSE >= 9:
                    os.remove(self.logfile_name)
            else:
                self.generate_setup()
                cmd = "python " + self.module + "_setup.py build_ext"
                if VERBOSE > 0: print "--- Instant: compiling ---"
                if VERBOSE > 1: print cmd
                ret, output = commands.getstatusoutput(cmd)
                compile_log_file = open("compile.log",  'w')
                compile_log_file.write(output)
                if ret != 0:
                    # compilation failed
                    os.remove("%s.md5" % self.module)
                    raise RuntimeError("The extension module did not compile, check %s/compile.log" % self.module)
                else:
                    #cmd = "python " + self.module + "_setup.py install --install-platlib=. >& compile.log 2>&1"
                    cmd = "python " + self.module + "_setup.py install --install-platlib=."
                    if VERBOSE > 1: print cmd
                    ret, output = commands.getstatusoutput(cmd)
                    compile_log_file.write(output)
                    if ret != 0:
                        # "installation" failed
                        os.remove("%s.md5" % self.module)
                        raise RuntimeError("Could not install the extension module, check %s/compile.log" % self.module)
        finally:
            # always get back to original directory
            os.chdir(previous_path)
            
            # Close the log file in case of a raised RuntimeError,
            # otherwise the stream will not get flushed
            compile_log_file.close()
        
        # Get md5 sum from .md5 file in temporary module dir
        tmp_module_dir = os.path.join(self.get_tmp_dir, self.module)
        file = open(os.path.join(tmp_module_dir, self.module + ".md5"))
        md5sum = file.readline()
        file.close()
        
        # Copy temporary module tree to cache
        if self.use_cache and md5sum:
            cache_module_dir = get_instant_module_dir(md5sum)
            shutil.copytree(tmp_module_dir, cache_module_dir)
            
            # Verify that everything is ok
            if VERBOSE > 9:
                print "Copying module tree to cache...", tmp_module_dir, cache_module_dir
            try:
                find_module(md5sum)
            except Exception, e:
                print "Failed to find module from checksum after compiling! Checksum is %s" % md5sum
                print e
    
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
        print 'generate_Interface',self.generate_Interface
        print 'signature',self.signature
        print 'use_cache'.self.use_cache
    
    def clean(self):
        """ Clean up files the current session. """
        if not gen_setup:
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
        for a in self.arrays:  
            # 1 dimentional arrays, ie. vectors
            if (len(a) == 2):  
                typemaps += """
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
""" % { 'n' : a[0] , 'array' : a[1] }
            # n dimentional arrays, ie. matrices and tensors  
            elif (len(a) == 3):  
                typemaps += """
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
""" % { 'n' : a[0] , 'ptv' : a[1], 'array' : a[2] }
            # end if
        # end for
        
        self.system_headers_code = "\n".join(['#include <%s>'  % header for header in self.system_headers])
        self.local_headers_code  = "\n".join(['#include "%s"'  % header for header in self.local_headers])
        self.wrap_headers_code1  = "\n".join(['#include "%s"'  % header for header in self.wrap_headers])
        self.wrap_headers_code2  = "\n".join(['%%include "%s"' % header for header in self.wrap_headers])

        self.typemaps = typemaps 

        interface_string = """
%%module  %(module)s
//%%module (directors="1") %(module)s

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

""" % vars(self)

        f = open(self.ifile_name, 'w')
        f.write(interface_string)
        f.close()
        if VERBOSE > 1:
            print '... Done'
        return func_name[func_name.rindex(' ')+1:func_name.index('(')]

    def check_md5sum(self): 
        """ 
        Check if the md5sum of the generated interface file has changed since the last
        time the module was compiled. If it has changed then recompilation is necessary.  
        """ 

        if self.signature: 
            current_md5sum = get_md5sum_from_signature(self.signature)
        else:
            md5sum_files = []
            md5sum_files.append(self.ifile_name)
            md5sum_files.extend(self.sources)
            md5sum_files.extend(self.wrap_headers)
            md5sum_files.extend(self.local_headers)
            current_md5sum = get_md5sum_from_files(md5sum_files)
            if VERBOSE > 2:
                print "md5sum_files ", md5sum_files
        
        if os.path.isfile(self.module+".md5"):
            if self.use_cache and find_module(current_md5sum):
                return 1
            else:
                last_md5sum = open(self.module + ".md5").readline()
                if current_md5sum == last_md5sum:
                    return 1
                else:
                    write_md5sum_file(current_md5sum, self.module + ".md5")
                    return 0
        else:
            write_md5sum_file(current_md5sum, self.module + ".md5")
            if find_module(current_md5sum):
                return 1
        
        return 0

    def generate_setup(self):
        """
        Generates a setup.py file
        """
        # handle arguments
        self.cppsrcs.append( "%s_wrap.cxx" % self.module ) # Martin: is it safe to just append to this here?
        
        compile_args = ""
        if len(self.cppargs) > 0:  
            compile_args = ", extra_compile_args=%s" % self.cppargs 

        link_args = ""
        if len(self.lddargs) > 0:  
            link_args = ", extra_link_args=%s" % self.lddargs 

        inc_dir = ""
        if len(self.local_headers) > 0:
            inc_dir = "-I.."
        
        # generate
        code = """ 
import os
from distutils.core import setup, Extension
name = '%s' 
swig_cmd ='swig -python %s %s %s'
os.system(swig_cmd)
sources = %s 
setup(name = '%s', 
      ext_modules = [Extension('_' + '%s', sources, 
                     include_dirs=%s, library_dirs=%s, 
                     libraries=%s %s %s)])  
        """ % (self.module, inc_dir, self.swigopts, self.ifile_name, 
               self.cppsrcs, 
               self.module, self.module, self.include_dirs, self.library_dirs, self.libraries, compile_args, link_args)
        # write
        f = open(self.module+'_setup.py', 'w')
        f.write(code)
        f.close()

    def generate_Makefile(self):
        """
        Generates a project dependent Makefile, which includes and
        uses SWIG's own Makefile to create an extension module of
        the supplied C/C++ code.
        """
        # generate
        code = """
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
           self.module)
        # end code
        
        # write
        f = open(self.makefile_name, 'w')
        f.write(code)
        f.close()
        
        if VERBOSE > 1:
            print 'Makefile', self.makefile_name, 'generated'
    
    ### End of class instant


# convert list values to string
def list2str(list):
    s = str(list)
    for c in ['[', ']', ',', '\'']:
        s = s.replace(c, '')
    return s


def find_module_by_signature(signature):
    return find_module(get_md5sum_from_signature(signature))


def import_module_by_signature(signature):
    module_name = find_module_by_signature(signature)
    if not module_name:
        raise RuntimeError("Couldn't find module with signature %s" % signature)
    instant_dir = get_instant_dir()
    if not instant_dir in sys.path:
        sys.path.insert(0, instant_dir)
    exec("import %s as imported_module" % module_name)
    return imported_module


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
           - B{additional_definitions}:
              - A list of additional definitions (typically needed for inheritance) 
           - B{additional_declarations}:
              - A list of additional declarations (typically needed for inheritance) 
           - B{signature}:
              - A signature string to identify the form instead of the source code.
    """ 
    ext = instant()
    ext.create_extension(**args)


def inline(c_code, **args_dict):
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
    args_dict["code"] = c_code 
    if not args_dict.has_key("module"):
        args_dict["module"] = "inline_ext" 
    try: 
        func = c_code[:c_code.index('(')]
        ret, func_name = func.split()
        ext.create_extension(**args_dict)
        exec("from inline_ext import %s as func_name"% func_name) 
        return func_name
    except: 
        ext.create_extension(**args_dict)
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
    args_dict["system_headers"] = args_dict.get("system_headers", []) + ["arrayobject.h"]
    args_dict["include_dirs"] = args_dict.get("include_dirs", []) + \
        [[sys.prefix + "/include/python" + sys.version[:3] + "/Numeric", 
          sys.prefix + "/include" + "/Numeric"][sys.platform=='win32'],
          "/usr/local/include/python" + sys.version[:3] +  "/Numeric"]
    args_dict["init_code"] =  args_dict.get("init_code", "") + "\nimport_array();\n"

    try: 
        ext = instant()
        func = c_code[:c_code.index('(')]
        ret, func_name = func.split()

        ext.create_extension(**args_dict)

        exec("from inline_ext_numeric import %s as func_name" % func_name)
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
    import numarray 
    inc_dir =  [numarray.numinclude.include_dir, 
                "/usr/local/include/python" + sys.version[:3] + "/numarray",
                "/usr/include/python" + sys.version[:3] + "/numarray" ] 

    if args_dict.has_key("system_headers"):  
        args_dict["system_headers"].append ("arrayobject.h")
    else: 
        args_dict["system_headers"] = ["arrayobject.h"]

    if args_dict.has_key("include_dirs"): 
        args_dict["include_dirs"].extend(inc_dir)
    else: 
        args_dict["include_dirs"] = inc_dir
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
        #print commands.getstatusoutput("pkg-config --exists %s " % pack)
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
            raise OSError("The pkg-config file %s does not exist" % pack)

    return (includes,flags,libs, libdirs) 
        

