"""This module contains helper functions for working with checksums."""

import hashlib
from .output import instant_assert, instant_debug, instant_error

def compute_checksum(text="", filenames=[]):
    """
    Get the checksum value of filename
    modified based on Python24\Tools\Scripts\md5.py
    """
    instant_assert(isinstance(text, str), "Expecting string.")
    instant_assert(isinstance(filenames, (list,tuple)), "Expecting sequence.")
    
    m = hashlib.new('sha1')
    if text:
        m.update(text)
    
    for filename in sorted(filenames): 
        instant_debug("Adding file '%s' to checksum." % filename)
        try:
            fp = open(filename, 'rb')
        except IOError as e:
            instant_error("Can't open file '%s': %s" % (filename, e))
        
        try:
            while 1:
                data = fp.read()
                if not data:
                    break
                m.update(data)
        except IOError as e:
            instant_error("I/O error reading '%s': %s" % (filename, e))
        
        fp.close() 
    
    return m.hexdigest().lower()


def _test():
    signature = "(Test signature)"
    files = ["signatures.py", "__init__.py"]
    print()
    print("Signature:", repr(signature))
    print("Checksum:", compute_checksum(signature, []))
    print()
    print("Files:", files)
    print("Checksum:", compute_checksum("", files))
    print()

if __name__ == "__main__":
    _test()

