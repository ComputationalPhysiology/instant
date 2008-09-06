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
from output import instant_warning, instant_assert, instant_debug
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
        instant_warning("In instant.import_module_directly: Failed to import module '%s' from '%s'." % (modulename, path))
        module = None
    finally:
        sys.path.pop(0)
    return module


_memory_cache = {}
def memory_cached_module(moduleid):
    "Returns the cached module if found."
    module = _memory_cache.get(moduleid, None)
    instant_debug("Returning '%s' from memory_cached_module(%r)." % (module, moduleid))
    return module


def place_module_in_memory_cache(moduleid, module):
    "Place a compiled module in cache with given id."
    _memory_cache[moduleid] = module
    instant_debug("Putting module '%s' in cache with key %r." % (module, moduleid))


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
    
    instant_assert(isinstance(moduleid, str), "In instant.find_module_location: Expecting moduleid to be string in find_module_location.")
    
    for modulename in moduleid_interpretations(moduleid):
        # Check in current directory
        if os.path.isdir(modulename):
            return os.getcwd(), modulename
        # Check in cache directory
        if os.path.isdir(os.path.join(cache_dir, modulename)):
            return cache_dir, modulename
    
    instant_debug("In instant.find_module_location: Didn't find module with moduleid %r" % moduleid)
    return (None, None)


def import_and_cache_module(path, modulename, moduleid, original_moduleid):
    module = import_module_directly(path, modulename)
    instant_assert(module is not None, "Failed to import module found in cache. Modulename: '%s'; Path: '%s'." % (modulename, path))
    place_module_in_memory_cache(moduleid, module)
    if original_moduleid is not moduleid:
        place_module_in_memory_cache(original_moduleid, module)
    return module


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
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    
    # Check memory cache first
    module = memory_cached_module(moduleid)
    if module:
        return module
    
    # Didn't find module in memory cache, getting
    # signature from moduleid if it isn't a string
    original_moduleid = moduleid
    if hasattr(moduleid, "signature"):
        moduleid = moduleid.signature()
        instant_debug("Got signature '%s' from moduleid." % moduleid)
        # Code copied from find_module_location (optimization since we know we have the signature)
        checksum = compute_checksum(moduleid)
        modulename = modulename_from_checksum(checksum)
        # Check in current directory
        if os.path.isdir(modulename):
            path = os.getcwd()
            return import_and_cache_module(os.getcwd(), modulename, moduleid, original_moduleid)
        # Check in cache directory
        if os.path.isdir(os.path.join(cache_dir, modulename)):
            path = cache_dir
            return import_and_cache_module(path, modulename, moduleid, original_moduleid)
    
    # Check possible disk locations
    path, modulename = find_module_location(moduleid, cache_dir)
    if modulename is not None:
        return import_and_cache_module(path, modulename, moduleid, original_moduleid)
    
    # All attempts failed
    instant_debug("In instant.import_module: Can't import module with moduleid %r using cache directory %r." \
                  % (moduleid, cache_dir))
    return None


def cached_modules(cache_dir=None):
    "Return a list with the names of all cached modules."
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    return os.listdir(cache_dir)

