"""This module contains helper functions for working with temp and cache directories."""

# Utilities for directory handling:

import os
import errno
import shutil
import tempfile
import time
from output import instant_debug, instant_assert

_tmp_dir = None
def get_temp_dir():
    """Return a temporary directory for the duration of this process.
    
    Multiple calls in the same process returns the same directory.
    Remember to call delete_temp_dir() before exiting."""
    global _tmp_dir
    if _tmp_dir is None:
        datestring = "%d-%d-%d-%02d-%02d" % time.localtime()[:5]
        suffix = datestring + "_instant"
        _tmp_dir = tempfile.mkdtemp(suffix)
        instant_debug("Created temp directory '%s'." % _tmp_dir)
    return _tmp_dir

def delete_temp_dir():
    """Delete the temporary directory created by get_temp_dir()."""
    global _tmp_dir
    if _tmp_dir and os.path.isdir(_tmp_dir):
        shutil.rmtree(_tmp_dir, ignore_errors=True)
    _tmp_dir = None

def get_instant_dir():
    "Return the default instant directory, creating it if necessary."
    # os.path.expanduser works for Windows, Linux, and Mac
    # In Windows, $HOME is os.environ['HOMEDRIVE'] + os.environ['HOMEPATH']
    instant_dir = os.path.join(os.path.expanduser("~"), ".instant")
    makedirs(instant_dir)
    return instant_dir

def get_default_cache_dir():
    "Return the default cache directory."
    if "INSTANT_CACHE_DIR" in os.environ:
        cache_dir = os.environ["INSTANT_CACHE_DIR"]
    else:
        cache_dir = os.path.join(get_instant_dir(), "cache")
    makedirs(cache_dir)
    return cache_dir

def get_default_error_dir():
    "Return the default error directory."
    if "INSTANT_ERROR_DIR" in os.environ:
        error_dir = os.environ["INSTANT_ERROR_DIR"]
    else:
        error_dir = os.path.join(get_instant_dir(), "error")
    makedirs(error_dir)
    return error_dir

def validate_cache_dir(cache_dir):
    if cache_dir is None:
        return get_default_cache_dir()
    instant_assert(isinstance(cache_dir, str), "Expecting cache_dir to be a string.")
    cache_dir = os.path.abspath(cache_dir)
    makedirs(cache_dir)
    return cache_dir

def makedirs(path):
    """
    Creates a directory (tree). If directory already excists it does nothing.
    """
    try:
        os.makedirs(path)
        instant_debug("In instant.makedirs: Creating directory %r" % path)
    except os.error, e:
        if e.errno != errno.EEXIST:
            raise

def _test():
    from output import set_logging_level
    set_logging_level("DEBUG")
    print "Temp dir:", get_temp_dir()
    print "Instant dir:", get_instant_dir()
    print "Default cache dir:", get_default_cache_dir()
    print "Default error dir:", get_default_error_dir()
    delete_temp_dir()
   
if __name__ == "__main__":
    _test()

