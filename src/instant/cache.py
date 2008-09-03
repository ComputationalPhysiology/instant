"""
Example operations:
- modulename = modulename_from_checksum(checksum)
- modulename = modulename_from_checksum(compute_checksum(signature))
- found = find_module_location(moduleid)
- module = import_module_directly(path, modulename)
- module = import_module(modulename)
- module = import_module(checksum)
- module = import_module(compute_checksum(signature))
- modules = cached_modules()
- modules = cached_modules(cache_dir)
"""

import os, sys
from output import instant_warning, instant_assert
from paths import get_default_cache_dir
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
        instant_warning("Failed to import module '%s' from '%s'." % (modulename, path))
        module = None
    finally:
        sys.path.pop(0)
    return module


_memory_cache = {}
def memory_cached_module(moduleid):
    "Returns the cached module if found."
    return _memory_cache.get(moduleid, None)


def place_module_in_memory_cache(moduleid, module):
    "Place a compiled module in cache with given id."
    _memory_cache[moduleid] = module


def moduleid_interpretations(moduleid):
    # Attempt to see moduleid as modulename
    yield moduleid
    # Attempt to see moduleid as checksum
    yield modulename_from_checksum(moduleid)
    # Attempt to see moduleid as signature
    yield modulename_from_checksum(compute_checksum(moduleid))


def find_module_location(moduleid, cache_dir=None):
    """Given a moduleid and an optional cache directory,
    return the matching path and modulename, or (None, none)."""
    # Use default cache directory if none supplied
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    
    instant_assert(isinstance(moduleid, str), "Expecting moduleid to be string in find_module_location.")
    
    for modulename in moduleid_interpretations(moduleid):
        # Check in current directory
        if os.path.isdir(modulename):
            return os.getcwd(), modulename
        # Check in cache directory
        if os.path.isdir(os.path.join(cache_dir, modulename)):
            return cache_dir, modulename
    
    instant_warning("Didn't find module with moduleid %r" % moduleid)
    return (None, None)


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
    
    # Check memory cache first
    module = memory_cached_module(moduleid)
    if module is not None:
        return module
    
    # Didn't find module in memory cache, getting
    # signature from moduleid if it isn't a string
    if not isinstance(moduleid, str):
        signature = moduleid.signature()
        # Code copied from find_module_location (optimization since we know we have the signature)
        checksum = compute_checksum(signature)
        modulename = modulename_from_checksum(checksum)
        # Check in current directory
        if os.path.isdir(modulename):
            module = import_module_directly(os.getcwd(), modulename)
            place_module_in_memory_cache(moduleid, module)
            return module
        # Check in cache directory
        if os.path.isdir(os.path.join(cache_dir, modulename)):
            module = import_module_directly(cache_dir, modulename)
            place_module_in_memory_cache(moduleid, module)
            return module
    
    # Check possible disk locations
    path, modulename = find_module_location(moduleid, cache_dir)
    if modulename is not None:
        module = import_module_directly(path, modulename)
        place_module_in_memory_cache(moduleid, module)
        return module
    
    # All attempts failed
    instant_warning("Can't import module with moduleid %r using cache directory %r." % (moduleid, cache_dir))
    return None


def cached_modules(cache_dir=None):
    "Return a list with the names of all cached modules."
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    return os.listdir(cache_dir)

