"""This module contains helper functions for working with temp and cache directories."""

# Utilities for directory handling:

import os
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
    _check_or_create(instant_dir, "instant")
    return instant_dir

def get_default_cache_dir():
    "Return the default cache directory."
    if "INSTANT_CACHE_DIR" in os.environ:
        cache_dir = os.environ["INSTANT_CACHE_DIR"]
    else:
        cache_dir = os.path.join(get_instant_dir(), "cache")
    _check_or_create(cache_dir, "cache")
    return cache_dir

def get_default_error_dir():
    "Return the default error directory."
    if "INSTANT_ERROR_DIR" in os.environ:
        error_dir = os.environ["INSTANT_ERROR_DIR"]
    else:
        error_dir = os.path.join(get_instant_dir(), "error")
    _check_or_create(error_dir, "error")
    return error_dir

def validate_cache_dir(cache_dir):
    if cache_dir is None:
        return get_default_cache_dir()
    instant_assert(isinstance(cache_dir, str), "Expecting cache_dir to be a string.")
    cache_dir = os.path.abspath(cache_dir)
    _check_or_create(cache_dir, "cache")
    return cache_dir

def _check_or_create(directory, label):
    if not os.path.isdir(directory):
        instant_debug("Creating %s directory '%s'." % (label, directory))
        os.mkdir(directory)

def _test():
    print "Temp dir:", get_temp_dir()
    print "Instant dir:", get_instant_dir()
    print "Default cache dir:", get_default_cache_dir()
    print "Default error dir:", get_default_error_dir()
    delete_temp_dir()
   
if __name__ == "__main__":
    _test()

