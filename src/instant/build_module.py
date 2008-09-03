
import os
import sys
import shutil
from itertools import chain
from subprocess import Popen, PIPE, STDOUT

# FIXME: Import only the official interface
from output import *
from config import header_and_libs_from_pkgconfig
from paths import *
from signatures import *
from cache import *
from codegeneration import *


# Taken from http://ivory.idyll.org/blog/mar-07/replacing-commands-with-subprocess
def get_status_output(cmd, input=None, cwd=None, env=None):
    pipe = Popen(cmd, shell=True, cwd=cwd, env=env, stdout=PIPE, stderr=STDOUT)

    (output, errout) = pipe.communicate(input=input)
    assert not errout

    status = pipe.returncode

    return (status, output)


def build_module(modulename=None, source_directory=".",
                     code="", init_code="",
                     additional_definitions="", additional_declarations="",
                     sources=[], wrap_headers=[],
                     local_headers=[], system_headers=[],
                     include_dirs=['.'], library_dirs=[], libraries=[],
                     swigargs=['-c++', '-fcompact', '-O', '-I.', '-small'], cppargs=['-O2'], lddargs=[],
                     object_files=[], arrays=[],
                     generate_interface=True, generate_setup=True, generate_makefile=False,
                     signature=None, cache_dir=None):
    """Generate and compile a module from C/C++ code using SWIG.
    
    Arguments: 
    ==========
       - B{modulename}:
          - The name you want for the module.
            If specified, the module will not be cached.
            If missing, a name will be constructed based on
            a checksum of the other arguments, and the module
            will be placed in the global cache. String.
       - B{source_directory}:
          - The directory where used supplied files reside.
       - B{code}:
          - A string containing C or C++ code to be compiled and wrapped.
       - B{init_code}:
          - Code that should be executed when the instant module is imported.
       - B{additional_definitions}:
          - A list of additional definitions (typically needed for inheritance).
       - B{additional_declarations}:
          - A list of additional declarations (typically needed for inheritance). 
       - B{sources}:
          - A list of source files to compile and link with the module.
       - B{wrap_headers}:
          - A list of local header files that should be wrapped by SWIG.
       - B{local_headers}:
          - A list of local header files required to compile the wrapped code.
       - B{system_headers}:
          - A list of system header files required to compile the wrapped code.
       - B{include_dirs}:
          - A list of directories to search for header files.
       - B{library_dirs}:
          - A list of directories to search for libraries (C{-l}).
       - B{libraries}:
          - A list of libraries needed by the instant module.
       - B{swigargs}:
          - List of arguments to swig, e.g. C{["-lpointers.i"]} to include the SWIG pointers.i library.
       - B{cppargs}:
          - List of arguments to the compiler, e.g. C{["-D", "-U"]}.
       - B{lddargs}:
          - List of arguments to the linker, e.g. C{["-D", "-U"]}.
       - B{object_files}:
          - If you want to compile the files yourself. NOT YET SUPPORTED. # TODO
       - B{arrays}:
          - A list of the C arrays to be made from NumPy arrays. # FIXME: Describe this correctly. Tests pass arrays of arrays of strings.
       - B{generate_interface}:
          - A bool to indicate if you want to generate the interface files.
       - B{generate_setup}:
          - A bool to indicate if you want to generate the setup.py file.
            By default, setup.py is used and not the Makefile.
       - B{generate_makefile}:
          - A bool to indicate if you want to generate the Makefile.
            If this is True, the Makefile is used instead of setup.py.
       - B{signature}:
          - A signature string to identify the form instead of the source code.
       - B{cache_dir}:
          - A directory to look for cached modules and place new ones.
            If missing, a default directory is used. Note that the module
            will not be cached if C{modulename} is specified.
            The cache directory should not be used for anything else.
    """
    # Store original directory to be able to restore later
    original_path = os.getcwd()
    
    # --- Validate arguments 
    
    def assert_is_str(x):
        instant_assert(isinstance(x, str), "Expecting string.")
    
    def assert_is_bool(x):
        instant_assert(isinstance(x, bool), "Expecting bool.")
    
    def assert_is_str_list(x):
        instant_assert(isinstance(x, (list, tuple)), "Expecting sequence.")
        instant_assert(all(isinstance(i, str) for i in x), "Expecting sequence of strings.")
    
    def strip_strings(x):
        assert_is_str_list(x)
        return [s.strip() for s in x]
    
    def arg_strings(x):
        if isinstance(x, str):
            x = x.split()
        return strip_strings(x)

    instant_assert(modulename is None or isinstance(modulename, str), "Expecting modulename to be string or None.")
    assert_is_str(source_directory)
    assert_is_str(code)
    assert_is_str(init_code)
    assert_is_str(additional_definitions)
    assert_is_str(additional_declarations)
    sources        = strip_strings(sources)
    wrap_headers   = strip_strings(wrap_headers)
    local_headers  = strip_strings(local_headers)
    system_headers = strip_strings(system_headers)
    include_dirs   = strip_strings(include_dirs)
    library_dirs   = strip_strings(library_dirs)
    libraries      = strip_strings(libraries)
    swigargs       = arg_strings(swigargs)
    cppargs        = arg_strings(cppargs)
    lddargs        = arg_strings(lddargs)
    object_files   = strip_strings(object_files)
    arrays         = [strip_strings(a) for a in arrays]
    assert_is_bool(generate_interface)
    assert_is_bool(generate_setup)
    assert_is_bool(generate_makefile)
    instant_assert(signature is None or isinstance(signature, str) or hasattr(signature, "signature"),
        "Expecting signature to be None, string, or a hashable object with a signature() function.")
    instant_assert(cache_dir is None or isinstance(cache_dir, str), "Expecting cache_dir to be string or None.")
    
    # --- Replace arguments with defaults if necessary
    
    source_directory = os.path.abspath(source_directory)
    
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    else:
        assert_is_str(cache_dir)
        cache_dir = os.path.abspath(cache_dir)
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
    
    signature_object = signature
    if signature is not None:
        # Check if module is in memory cache
        module = memory_cached_module(signature)
        if module is not None:
            return module
        # Module not found in memory cache, make signature a string if it isn't
        if not isinstance(signature, str):
            signature = signature.signature()
            # FIXME: Must use signature_object as the key to place a module in the memory cache after compilation.
    
    # Split sources by file-suffix (.c or .cpp)
    csrcs = []
    cppsrcs = []
    for f in sources:
        if f.endswith('.cpp') or f.endswith('.cxx'):
            cppsrcs.append(f)
        elif f.endswith('.c') or f.endswith('.C'):
            csrcs.append(f)
        else:
            instant_error("Source files must have '.c' or '.cpp' suffix, this is not the case for '%s'." % f)

    # --- Debugging code
    instant_debug('::: Begin Arguments :::')
    instant_debug('    modulename: %r' % modulename)
    instant_debug('    code: %r' % code)
    instant_debug('    init_code: %r' % init_code)
    instant_debug('    additional_definitions: %r' % additional_definitions)
    instant_debug('    additional_declarations: %r' % additional_declarations)
    instant_debug('    sources: %r' % sources)
    instant_debug('    csrcs: %r' % csrcs)
    instant_debug('    cppsrcs: %r' % cppsrcs)
    instant_debug('    wrap_headers: %r' % wrap_headers)
    instant_debug('    local_headers: %r' % local_headers)
    instant_debug('    system_headers: %r' % system_headers)
    instant_debug('    include_dirs: %r' % include_dirs)
    instant_debug('    library_dirs: %r' % library_dirs)
    instant_debug('    libraries: %r' % libraries)
    instant_debug('    swigargs: %r' % swigargs)
    instant_debug('    cppargs: %r' % cppargs)
    instant_debug('    lddargs: %r' % lddargs)
    instant_debug('    object_files: %r' % object_files)
    instant_debug('    arrays: %r' % arrays)
    instant_debug('    generate_interface: %r' % generate_interface)
    instant_debug('    generate_setup: %r' % generate_setup)
    instant_debug('    generate_makefile: %r' % generate_makefile)
    instant_debug('    signature: %r' % signature)
    instant_debug('    cache_dir: %r' % cache_dir)
    instant_debug('::: End Arguments :::')

    # --- Wrapping rest of code in try-block to clean up at the end if something fails.
    try:  
        # --- Setup module directory, making it and copying files to it if necessary
        
        # Create module path where the module is to be built,
        # either in a local directory if a module name is given,
        # or in a temporary directory for later copying to cache.
        if modulename:
            use_cache = False
            module_path = os.path.join(original_path, modulename)
        else:
            use_cache = True
            # Compute cache_checksum (this is _before_ interface files are generated!)
            if signature is None:
                # TODO: Add all files and arguments we want here!
                allfiles = sources + wrap_headers + local_headers
                text = code + init_code + additional_definitions + additional_declarations
                cache_checksum = compute_checksum(text, allfiles)
            else:
                # If given a user-provided signature, we don't look at anything else.
                cache_checksum = compute_checksum(signature, [])
            # Lookup cache_checksum in cache
            cached_module = import_module(cache_checksum, cache_dir)
            if cached_module is not None:
                instant_debug("Found module '%s' in cache." % cached_module.__name__)
                return cached_module
            # Define modulename and path automatically
            modulename = modulename_from_checksum(cache_checksum)
            module_path = os.path.join(get_temp_dir(), modulename)
            instant_assert(not os.path.exists(module_path), "")
        
        # --- Copy files to module path
        module_path = os.path.abspath(module_path)
        if os.path.exists(module_path):
            instant_warning("Path '%s' already exists, may overwrite existing files." % module_path)
        else:
            os.mkdir(module_path)
        
        # Copy source files to module_path if necessary
        if source_directory != module_path:
            # Create list of files that should be copied
            files_to_copy = []
            files_to_copy.extend(sources) 
            files_to_copy.extend(wrap_headers)
            files_to_copy.extend(local_headers)
            files_to_copy.extend(object_files)
            
            instant_debug("Copying files %r from %r to %r" % (files_to_copy, source_directory, module_path))
            for f in files_to_copy:
                a = os.path.join(source_directory, f)
                b = os.path.join(module_path, f)
                if a == b:
                    instant_debug("If this prints, I'd like to know how it happened!")
                    continue
                instant_assert(os.path.isfile(a), "Missing file '%s'." % a)
                if os.path.isfile(b):
                    instant_warning("Overwriting file '%s' with '%s'." % (b, a))
                    os.remove(b)
                shutil.copyfile(a, b)
        
        # At this point, all user input files should reside in module_path.
        
        # --- Generate files in module directory
        
        os.chdir(module_path)
        
        # Generate __init__.py which imports compiled module contents
        write_file("__init__.py", "from %s import *" % modulename)
        
        # Generate SWIG interface, setup.py, and Makefile if wanted
        ifile_name = "%s.i" % modulename
        if generate_interface:
            ifile_name2 = write_interfacefile(modulename, code, init_code, additional_definitions, additional_declarations, system_headers, local_headers, wrap_headers, arrays)
            instant_assert(ifile_name == ifile_name2, "Logic breach in build_module, %r != %r." % (ifile_name, ifile_name2))
        
        if generate_setup:
            setup_name = write_setup(modulename, csrcs, cppsrcs, local_headers, include_dirs, library_dirs, libraries, swigargs, cppargs, lddargs)
        
        if generate_makefile:
            makefile_name = write_makefile(modulename, csrcs, cppsrcs, local_headers, include_dirs, library_dirs, libraries, swigargs, cppargs, lddargs)
        
        # --- Build module
        # At this point we have all the files, and can make the
        # total checksum from all file contents. This is used to
        # decide wether the module needs recompilation or not.
        
        # Compute new_compilation_checksum
        allfiles = sources + wrap_headers + local_headers + [ifile_name]
        text = "" # TODO: Maybe append *args here? (all sourcecode text is embedded in above files)
        new_compilation_checksum = compute_checksum(text, allfiles)
        
        compilation_checksum_filename = "%s.checksum" % modulename
        
        # Check if the old checksum matches the new one
        need_recompilation = True
        if os.path.exists(compilation_checksum_filename):
            checksum_file = open(compilation_checksum_filename)
            old_compilation_checksum = checksum_file.readline()
            checksum_file.close()
            if old_compilation_checksum == new_compilation_checksum:
                need_recompilation = False
        
        if need_recompilation:
            # Verify that SWIG is on the system
            (swig_stat, swig_out) = get_status_output("swig -version")
            if swig_stat != 0:
                instant_error("Could not find swig! You can download swig from http://www.swig.org")
            
            # Create log file for logging of compilation errors
            compile_log_filename = os.path.join(module_path, "compile.log")
            compile_log_file = open(compile_log_filename, "w")
            
            # Run makefile or setup.py.
            # The default is setup.py, so if the user
            # told us to make a makefile, we use that.
            if generate_makefile:
                # clean module
                cmd = "make -f %s clean" %  makefile_name
                instant_info("--- Instant: compiling ---")
                instant_info(cmd)
                ret, output = get_status_output(cmd)
                compile_log_file.write(output)
                compile_log_file.flush()
                if ret != 0:
                    os.remove(compilation_checksum_filename)
                    instant_error("Failed cleaning the module directory, see '%s'" % compile_log_filename)
                
                # build module
                cmd = "make -f %s python" % makefile_name
                instant_info(cmd)
                ret, output = get_status_output(cmd)
                compile_log_file.write(output)
                compile_log_file.flush()
                if ret != 0:
                    os.remove(compilation_checksum_filename)
                    instant_error("The module did not compile, see '%s'" % compile_log_filename)
            else:
                # build module
                cmd = "python %s build_ext" % setup_name
                instant_info("--- Instant: compiling ---")
                instant_info(cmd)
                ret, output = get_status_output(cmd)
                compile_log_file.write(output)
                compile_log_file.flush()
                if ret != 0:
                    os.remove(compilation_checksum_filename)
                    instant_error("The module did not compile, see '%s'" % compile_log_filename)
                
                # 'install' module
                cmd = "python %s install --install-platlib=." % setup_name
                instant_info(cmd)
                ret, output = get_status_output(cmd)
                compile_log_file.write(output)
                compile_log_file.flush()
                if ret != 0:
                    os.remove(compilation_checksum_filename)
                    instant_error("Could not 'install' the module, see '%s'" % compile_log_filename)
            
            # Compilation succeeded, write new_compilation_checksum to checksum_file
            write_file(compilation_checksum_filename, new_compilation_checksum)
        
        # --- Load module and return it
        if use_cache:
            # Copy compiled module to cache
            cache_module_path = os.path.join(cache_dir, modulename)
            if os.path.exists(cache_module_path):
                instant_warning("Path '%s' already exists, but module wasn't found in cache previously. Overwriting." % cache_module_path) # TODO: Error instead? Indicates race condition on disk or bug in Instant.
                shutil.rmtree(cache_module_path)
            instant_info("Copying built module from %r to cache at %r" % (module_path, cache_module_path))
            instant_assert(os.path.isdir(module_path), "Cannot copy non-existing directory %r!" % module_path)
            instant_assert(not os.path.isdir(cache_module_path), "Cache directory %r shouldn't exist at this point!" % cache_module_path)
            shutil.copytree(module_path, cache_module_path)
            delete_temp_dir()
            # Verify that we can load the module from the cache now
            cached_module = import_module(cache_checksum, cache_dir)
            if cached_module is None:
                instant_error("Failed to find module in cache from checksum after compiling! Checksum is '%s'" % cache_checksum)
            instant_debug("Returning module '%s' from build_module." % cached_module.__name__)
            return cached_module
        else:
            compiled_module = import_module_directly(module_path, modulename)
            instant_debug("Returning %s from build_module." % compiled_module)
            return compiled_module
        
        # The end!
    finally:
        # Always get back to original directory.
        os.chdir(original_path)
        
        # Close the log file in case we were aborted.
        compile_log_file = locals().get("compile_log_file", None)
        if compile_log_file:
            compile_log_file.close()
    
    instant_error("Should never reach this point!")
    # end build_module

