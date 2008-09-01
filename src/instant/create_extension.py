

import os
import sys
import shutil
import commands 

# FIXME: Import only the official interface
from output import *
from config import header_and_libs_from_pkgconfig
from paths import *
from signatures import *
from cache import *
from codegeneration import *
from create_extension import *
from highlevel import *


# FIXME order:
# - parse_args
# - running setup
# - check_md5


def parse_args(kwargs): # FIXME: Create argument class, populate it here, and return it
    """Parse kwargs dict and return a validated."""

    args = InstantConfig()

    # FIXME: Update below code...
    self.code = reindent("""
        void f()
        {
            printf("No code supplied!\\n");
        }
        """)
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
    self.generate_interface = True
    self.signature          = ""
    self.use_cache          = False

    for key in kwargs.keys():
        if key == 'code':
            self.code = kwargs[key]
        elif key == 'module':
            self.module = kwargs[key]
        elif key == 'swigopts':
            self.swigopts = kwargs[key]
        elif key == 'init_code':
            self.init_code = kwargs[key]
        elif key == 'sources':
            self.sources = kwargs[key]
        elif key == 'system_headers':
            self.system_headers = kwargs[key]
        elif key == 'local_headers':
            self.local_headers = kwargs[key]
            self.include_dirs.append("..")
        elif key == 'wrap_headers':
            self.wrap_headers = kwargs[key]
        elif key == 'include_dirs':
            self.include_dirs.extend(kwargs[key])
        elif key == 'libraries':
            self.libraries = kwargs[key]
        elif key == 'library_dirs':
            self.library_dirs = kwargs[key]
        elif key == 'cppargs':
            assert isinstance(kwargs[key], (str,tuple, list)), "Wrong type of argument to cppargs" 
            if isinstance(kwargs[key], str):
                if kwargs[key] is "":
                    self.cppargs = []
                else:
                    self.cppargs = [kwargs[key].strip()]
            elif isinstance(kwargs[key], (tuple, list)):
                self.cppargs = [s.strip() for s in kwargs[key]] 
        elif key == 'lddargs':
            assert isinstance(kwargs[key], (str,tuple, list)), "Wrong type of argument to lddargs" 
            if isinstance(kwargs[key], str): 
                if kwargs[key] is "":
                    self.lddargs = []
                else:
                    self.lddargs = [kwargs[key].strip()]
            elif isinstance(kwargs[key], (tuple, list)):
                self.lddargs = [s.strip() for s in kwargs[key]] 
        elif key == 'object_files':
            self.object_files = kwargs[key]
        elif key == 'arrays':
            self.arrays = kwargs[key]
        elif key == 'additional_definitions':
            self.additional_definitions = kwargs[key]
        elif key == 'additional_declarations':
            self.additional_declarations = kwargs[key]
        elif key == 'generate_interface': 
            self.generate_interface= kwargs[key]
        elif key == 'signature': 
            self.signature = kwargs[key]
        elif key == 'use_cache':
            self.use_cache = kwargs[key]

    if isinstance(self.use_cache, bool) or isinstance(self.use_cache, int):
        self.use_cache = get_instant_dir()
    elif isinstance(self.use_cache, str) and self.use_cache:
        self.use_cache = os.path.join(os.getcwd(), self.use_cache)

    global USE_CACHE
    USE_CACHE = self.use_cache

    self.makefile_name = self.module+".mak"
    self.logfile_name  = self.module+".log"
    self.ifile_name    = self.module+".i"
    # sources has to be put in different variables,
    # depending on the file-suffix (.c or .cpp).
    self.csrcs = []
    self.cppsrcs = []
    for file in self.sources:
        if file.endswith('.cpp'):
            self.cppsrcs.append(file)
        elif file.endswith('.c'):
            self.csrcs.append(file)
        else:
            instant_error("Source files must have '.c' or '.cpp' suffix, This is not the case for", file)
    
    return args 


def run_command(cmd):
    "Run a command, assert success, and return output."
    ret, output = commands.getstatusoutput(cmd)
    instant_assert(ret == 0, "Failed to run command '%s'" % cmd)
    return output


# FIXME: Where to use this?
def check_md5sum(self): 
    """ 
    Check if the md5sum of the generated interface file has changed since the last
    time the module was compiled. If it has changed then recompilation is necessary.  
    """ 
    if self.signature: 
        current_md5sum = get_md5sum_from_signature(self.signature)
        md5_filename = self.module + ".md5"
        write_file(md5_filename, current_md5sum)
    else:
        md5sum_files = []
        md5sum_files.append(self.ifile_name)
        md5sum_files.extend(self.sources)
        md5sum_files.extend(self.wrap_headers)
        md5sum_files.extend(self.local_headers)
        current_md5sum = get_md5sum_from_files(md5sum_files)
        instant_debug("md5sum_files ", md5sum_files)

    if current_md5sum is not None:
        md5_filename = os.path.join(get_tmp_dir(), self.module + ".md5")
        write_file(md5_filename, current_md5sum)

    if os.path.isfile(self.module+".md5"):
        if len(self.use_cache) > 1 and find_module(current_md5sum):
            return 1
        else:
            last_md5sum = open(self.module + ".md5").readline()
            if current_md5sum == last_md5sum:
                return 1
            else:
                md5_filename = self.module + ".md5"
                write_file(md5_filename, current_md5sum)
                return 0
    else:
        if find_module(current_md5sum):
            return 1
    
    return 0


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
       - B{generate_interface}:
          - Indicate if you want to generate the interface files. Bool.
       - B{signature}:
          - A signature string to identify the form instead of the source code.
       - B{use_cache}:
          - Indicate if you want to store the generated module for later use.
            If a bool or int is given that is evaluated as True, the
            default cache directory will be used. To specify a cache
            directory, a string with with the path relative to the current path is used.
    """
    
    previous_path = os.getcwd()
    
    # create module path, either in cache or a local directory
    if len(self.use_cache) > 1: 
        module_path = os.path.join(get_instant_dir(), self.module) 
    else: 
        module_path = self.module
    if not os.path.isdir(module_path): 
        os.mkdir(module_path)
    
    # copy files either to cache or to local directory
    # create list of files that should be copied
    files_to_copy = []
    files_to_copy.extend(self.sources) 
    files_to_copy.extend(self.local_headers)
    files_to_copy.extend(self.object_files)
    files_to_copy.extend(self.wrap_headers)

    instant_debug("Copying files: ", files_to_copy, " to ", module_path)
    for file in files_to_copy:
        if not os.path.isfile(os.path.join(module_path, file)):
            shutil.copyfile(file, os.path.join(module_path, file))

    # generate __init__.py which imports compiled module contents
    os.chdir(module_path)
    f = open("__init__.py", 'w')
    f.write("from %s import *"% self.module)
    f.close()
    
    # generate interface files if wanted
    if self.generate_interface:
        self.generate_interfacefile()
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

        # open file already here, else we might get an UnBoundLocalError 
        compile_log_file = open("compile.log",  'w')
        # generate Makefile or setup.py and run it
        if not self.gen_setup:
            self.generate_makefile()
            run_command("make -f %s clean" %  self.makefile_name)
            output = run_command("make -f %s" % self.makefile_name)
            write_file(self.logfile_name, output)
        else:
            self.generate_setup()
            cmd = "python " + self.module + "_setup.py build_ext"
            instant_info("--- Instant: compiling ---")
            instant_info(cmd)
            ret, output = commands.getstatusoutput(cmd)
            compile_log_file.write(output)
            if ret != 0:
                # compilation failed
                os.remove("%s.md5" % self.module)
                raise RuntimeError("The extension module did not compile, check %s/compile.log" % self.module)
            else:
                #cmd = "python " + self.module + "_setup.py install --install-platlib=. >& compile.log 2>&1"
                cmd = "python " + self.module + "_setup.py install --install-platlib=."
                instant_info(cmd)
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
    tmp_module_dir = get_tmp_dir()
    file = open(os.path.join(tmp_module_dir, self.module + ".md5"))
    md5sum = file.readline()
    file.close()
    
    # Copy temporary module tree to cache
    if len(self.use_cache) > 1 and md5sum:
        cache_module_dir = os.path.join(get_instant_dir(), "instant_module_" + md5sum)
        shutil.copytree(tmp_module_dir, cache_module_dir)
        
        # Verify that everything is ok
        instant_info("Copying module tree to cache...", tmp_module_dir, cache_module_dir)
        cached_module = find_module(md5sum)
        if cached_module is None:
            instant_warning("Failed to find module from checksum after compiling! Checksum is %s" % md5sum)


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
           - B{generate_interface}:
              - Indicate if you want to generate the interface files. Bool.
           - B{signature}:
              - A signature string to identify the form instead of the source code.
           - B{use_cache}:
              - Indicate if you want to store the generated module for later use.
                If a bool or int is given that is evaluated as True, the
                default cache directory will be used. To specify a cache
                directory, a string with with the path relative to the current path is used.
    """ 

    # FIXME: Paths are fucked up, use this structure:
    ### Path structure:
    # temp_dir = given by tempfile, should be deleted when closing
    # instant_dir = persistent directory in the current users home
    # cache_dir = pjoin(instant_dir, "cache")
    # module_parent_dir = cache_dir, if you want it locally set cache_dir="."
    # module_name (module files are placed under pjoin(module_parent_dir, module_name), module module_name is imported from module_parent_dir)

    args = parse_args(kwargs)
    
    sig = make_signature(args)
    
    if extension_in_cache(sig):
        return import_extension(sig)
    
    config = make_config(args)
    
    generate_interfacefile(args, config)
    
    if use_makefile:
        makefile = generate_makefile(args, config)
        run_command("make -f %s clean" % makefile) # FIXME: fix arguments
        run_command("make -f %s all" % makefile) # FIXME: fix arguments
    else:
        setup = generate_setup(args, config)
        run_command("%s build" % setup) # FIXME: fix arguments
        run_command("%s install" % setup) # FIXME: fix arguments
    
    return import_extension(sig)

