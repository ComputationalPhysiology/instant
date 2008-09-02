"""
Example operations:
- modulename = modulename_from_md5sum(md5sum)
- modulename = modulename_from_md5sum(compute_md5(signature))
- found = in_cache(md5sum)
- found = in_cache(compute_md5(signature))
- module = import_extension_from_cache(md5sum)
- module = import_extension_from_cache(compute_md5(signature))
- module = import_extension(path, modulename)
- modules = cached_extensions()
- modules = cached_extensions(path)
"""

import os, sys
from output import instant_warning, instant_assert
from paths import get_default_cache_dir
from signatures import compute_md5


# TODO: We could make this an argument, but it's used indirectly several places so take care.
_modulename_prefix = "instant_module_"


def modulename_from_md5sum(md5sum):
    "Construct a module name from a md5 sum for use in cache."
    return _modulename_prefix + md5sum


def md5sum_from_modulename(modulename):
    "Construct a module name from a md5 sum for use in cache."
    return modulename.remove(_modulename_prefix)


def find_extension(signature, cache_dir=None):
    "Return wether a module with the given signature or md5sum is found in cache."
    # TODO: This does a rather crude check, we could check some directory contents as well to make it more robust.
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    
    # Attempt to see signature as md5sum
    modulename = modulename_from_md5sum(signature)
    if os.path.isdir(os.path.join(cache_dir, modulename)):
        return True
    
    # Compute md5sum of signature
    signature = compute_md5(signature)
    modulename = modulename_from_md5sum(signature)
    if os.path.isdir(os.path.join(cache_dir, modulename)):
        return True
    
    # All attempts failed
    return False


def import_extension_directly(path, modulename):
    "Import an extension module with the given module name that resides in the given path."
    sys.path.insert(0, path)
    try:
        extension = __import__(modulename)
    except:
        instant_warning("Failed to import extension module '%s' from '%s'." % (modulename, path))
        extension = None
    finally:
        sys.path.pop(0)
    return extension


def import_extension(signature, cache_dir=None):
    "Import extension from cache given its signature or md5sum."
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    
    # Attempt to see signature as md5sum
    modulename = modulename_from_md5sum(signature)
    if os.path.isdir(os.path.join(cache_dir, modulename)):
        return import_extension_directly(cache_dir, modulename)
    
    # Compute md5sum of signature
    signature = compute_md5(signature)
    modulename = modulename_from_md5sum(signature)
    if os.path.isdir(os.path.join(cache_dir, modulename)):
        return import_extension_directly(cache_dir, modulename)

    # All attempts failed.
    instant_warning("Can't find module with signature or md5sum '%s' in cache at '%s'." % (signature, cache_dir))
    return None


def cached_extensions(cache_dir=None):
    "Return a list with the names of all cached extension modules."
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    return os.listdir(cache_dir)

