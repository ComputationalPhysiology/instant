

def get_func_name(code): # This hack was part of generate_interfacefile, but unused.
    func_name = code[:code.index(')')+1]
    return func_name[func_name.rindex(' ')+1:func_name.index('(')]


def clean(self): # FIXME: Unused, needed anywhere?
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


def debug(self): # FIXME: Merge with create_extension
    """Print out all instance variables."""
    instant_debug('DEBUG CODE:')
    instant_debug('code',self.code)
    instant_debug('module',self.module)
    instant_debug('swigopts',self.swigopts)
    instant_debug('init_code',self.init_code)
    instant_debug('system_headers',self.system_headers)
    instant_debug('local_headers',self.local_headers)
    instant_debug('wrap_headers',self.wrap_headers)
    instant_debug('include_dirs',self.include_dirs)
    instant_debug('sources',self.sources)
    instant_debug('srcs',self.srcs)
    instant_debug('cppsrcs',self.cppsrcs)
    instant_debug('cppargs',self.cppargs)
    instant_debug('lddargs',self.lddargs)
    instant_debug('libraries',self.libraries)
    instant_debug('library_dirs',self.library_dirs)
    instant_debug('object_files',self.object_files)
    instant_debug('arrays',self.arrays)
    instant_debug('additional_definitions',self.additional_definitions)
    instant_debug('additional_declarations',self.additional_declarations)
    instant_debug('generate_interface',self.generate_interface)
    instant_debug('signature',self.signature)
    instant_debug('use_cache',self.use_cache)


def _path_walk_callback(arg, directory, files):
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
    instant_dir = os.path.abspath(get_instant_dir())
    os.path.walk(instant_dir, _path_walk_callback, arg)
    if len(arg) == 2:
        assert arg[1]
        directory = os.path.join(get_instant_module_dir(md5sum), arg[1])
        instant_debug("find_module: directory = ", directory)
        # add found module directory to path
        if not directory in sys.path:
            instant_debug("Inserting directory in sys.path: ", directory)
            sys.path.insert(0, directory) 
        # return module (directory) name
        instant_debug("find_module returning:", os.path.split(arg[1])[-1])
        return os.path.split(arg[1])[-1]
        #return 1
    return None


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

