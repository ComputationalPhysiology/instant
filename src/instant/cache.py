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

import os
from paths import get_default_cache_dir


# TODO: Could make this an argument, but it's used indirectly many places.
_modulename_prefix = "instant_module_"


def modulename_from_md5sum(md5sum):
    "Construct a module name from a md5 sum for use in cache."
    return _modulename_prefix + md5sum


def md5sum_from_modulename(modulename):
    "Construct a module name from a md5 sum for use in cache."
    return modulename.remove(_modulename_prefix)


def in_cache(md5sum, cache_dir=None):
    "Return wether a module with the given md5 sum is found in cache."
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    modulename = modulename_from_md5sum(md5sum)
    if os.path.isdir(os.path.join(cache_dir, modulename)):
        # TODO: A rather crude check, check directory contents as well?
        return True
    return False


def import_extension(path, modulename):
    "Import an extension module with the given module name that resides in the given path."
    sys.path.insert(0, path)
    try:
        extension = __import__(modulename)
    except:
        instant_warning("Failed to import extension module '%s' from '%s'." % (modulename, path))
        extension = None
    finally:
        sys.path.remove(0)
    return extension


def import_extension_from_cache(md5sum, cache_dir=None):
    "Import extension from cache given its md5sum."
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    instant_assert(in_cache(md5sum, cache_dir),
        "Can't find module with md5sum '%s' in cache at '%s'." % (md5sum, cache_dir))
    modulename = modulename_from_md5sum(md5sum)
    return import_extension(cache_dir, modulename)


def cached_extensions(cache_dir=None):
    "Return a list with the names of all cached extension modules."
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    return os.list(cache_dir)

