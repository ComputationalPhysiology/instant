
# Utilities for directory handling:

import os
import shutil
import tempfile
import time
from output import instant_debug

_tmp_dir = None
def get_temp_dir():
    """Return a temporary directory for the duration of this process.
    
    Multiple calls in the same process returns the same directory.
    Remember to all delete_temp_dir() before exiting."""
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
    "Return a temporary directory for the duration of this process."
    # os.path.expanduser works for Windows, Linux, and Mac
    # In Windows, $HOME is os.environ['HOMEDRIVE'] + os.environ['HOMEPATH']
    instant_dir = os.path.join(os.path.expanduser("~"), ".instant")
    if not os.path.isdir(instant_dir):
        instant_debug("Creating instant directory '%s'." % instant_dir)
        os.mkdir(instant_dir)
    return instant_dir

def get_default_cache_dir():
    "Return the default cache directory."
    cache_dir = os.path.join(get_instant_dir(), "cache")
    if not os.path.isdir(cache_dir):
        instant_debug("Creating cache directory '%s'." % cache_dir)
        os.mkdir(cache_dir)
    return cache_dir

def validate_cache_dir(cache_dir):
    if cache_dir is None:
        return get_default_cache_dir()
    assert_is_str(cache_dir)
    cache_dir = os.path.abspath(cache_dir)
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    return cache_dir

def _test():
    print "Temp dir:", get_temp_dir()
    print "Instant dir:", get_instant_dir()
    print "Default cache dir:", get_default_cache_dir()
    delete_temp_dir()
   
if __name__ == "__main__":
    _test()

