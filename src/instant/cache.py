"""
Example operations:
- modulename = modulename_from_checksum(checksum)
- modulename = modulename_from_checksum(compute_checksum(signature))
- module = import_module_directly(path, modulename)
- module = import_module(modulename)
- module = import_module(checksum)
- module = import_module(compute_checksum(signature))
- modules = cached_modules()
- modules = cached_modules(cache_dir)
"""

import os, sys, re
from output import instant_warning, instant_assert, instant_debug
from paths import get_default_cache_dir, validate_cache_dir
from signatures import compute_checksum


# TODO: We could make this an argument, but it's used indirectly several places so take care.
_modulename_prefix = "instant_module_"
def modulename_from_checksum(checksum):
    "Construct a module name from a checksum for use in cache."
    return _modulename_prefix + checksum


def checksum_from_modulename(modulename):
    "Construct a module name from a checksum for use in cache."
    return modulename.remove(_modulename_prefix)


def import_module_directly(path, modulename):
    "Import a module with the given module name that resides in the given path."
    sys.path.insert(0, path)
    try:
        module = __import__(modulename)
    except:
        instant_warning("In instant.import_module_directly: Failed to import module '%s' from '%s'." % (modulename, path))
        module = None
    finally:
        sys.path.pop(0)
    return module


_memory_cache = {}
def memory_cached_module(moduleid):
    "Returns the cached module if found."
    module = _memory_cache.get(moduleid, None)
    instant_debug("Found '%s' in memory cache with key '%r'." % (module, moduleid))
    return module


def place_module_in_memory_cache(moduleid, module):
    "Place a compiled module in cache with given id."
    _memory_cache[moduleid] = module
    instant_debug("Added module '%s' to cache with key '%r'." % (module, moduleid))


def is_valid_module_name(name):
    NAMELENGTHLIMIT = 100
    return len(name) < NAMELENGTHLIMIT and bool(re.search(r"^[a-zA-Z_][\w]*$", name))


def import_and_cache_module(path, modulename, moduleids):
    module = import_module_directly(path, modulename)
    instant_assert(module is not None, "Failed to import module found in cache. Modulename: '%s'; Path: '%s'." % (modulename, path))
    for moduleid in moduleids:
        place_module_in_memory_cache(moduleid, module)
    return module


def check_memory_cache(moduleid):
    # Check memory cache first with the given moduleid
    moduleids = [moduleid]
    module = memory_cached_module(moduleid)
    if module: return module, moduleids
    
    # Get signature from moduleid if it isn't a string,
    # and check memory cache again
    if hasattr(moduleid, "signature"):
        moduleid = moduleid.signature()
        instant_debug("In instant.check_memory_cache: Got signature "\
                      "'%s' from moduleid.signature()." % moduleid)
        module = memory_cached_module(moduleid)
        if module:
            for moduleid in moduleids:
                place_module_in_memory_cache(moduleid, module)
            return module, moduleids
        moduleids.append(moduleid)
    
    # Construct a filename from the checksum of moduleid if it
    # isn't already a valid name, and check memory cache again
    if not is_valid_module_name(moduleid):
        moduleid = modulename_from_checksum(compute_checksum(moduleid))
        instant_debug("In instant.check_memory_cache: Constructed module name "\
                      "'%s' from moduleid '%s'." % (moduleid, moduleids[-1]))
        module = memory_cached_module(moduleid)
        if module: return module, moduleids
        moduleids.append(moduleid)
    
    instant_debug("In instant.check_memory_cache: Failed to find module.")
    return None, moduleids


def check_disk_cache(modulename, cache_dir, moduleids):
    # Ensure a valid cache_dir
    cache_dir = validate_cache_dir(cache_dir)
    
    # Check on disk, in current directory and cache directory
    for path in (os.getcwd(), cache_dir):
        if os.path.isdir(os.path.join(path, modulename)):
            # Found existing directory, try to import and place in memory cache
            module = import_and_cache_module(path, modulename, moduleids)
            if module:
                instant_debug("In instant.check_disk_cache: Imported module "\
                              "'%s' from '%s'." % (modulename, path))
                return module
            else:
                instant_debug("In instant.check_disk_cache: Failed to imported "\
                              "module '%s' from '%s'." % (modulename, path))
    
    # All attempts failed
    instant_debug("In instant.check_disk_cache: Can't import module with modulename "\
                  "%r using cache directory %r." % (modulename, cache_dir))
    return None


def import_module(moduleid, cache_dir=None):
    """Import module from cache given its moduleid and an optional cache directory.
    
    The moduleid can be either
    - the module name
    - a signature string, of which a checksum is taken to look up in the cache
    - a checksum string, which is used directly to look up in the cache
    - a hashable non-string object with a function moduleid.signature() which is used to get a signature string
    The hashable object is used to look up in the memory cache before signature() is called.
    If the module is found on disk, it is placed in the memory cache.
    """
    # Look for module in memory cache
    module, moduleids = check_memory_cache(moduleid)
    if module: return module
    
    # Look for module in disk cache
    modulename = moduleids[-1]
    return check_disk_cache(modulename, cache_dir, moduleids)


def cached_modules(cache_dir=None):
    "Return a list with the names of all cached modules."
    cache_dir = validate_cache_dir(cache_dir)
    return os.listdir(cache_dir)

