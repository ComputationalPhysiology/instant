
import hashlib
from output import instant_assert, instant_debug, instant_error

def compute_md5(text="", filenames=[]):
    """
    Get the md5 value of filename
    modified based on Python24\Tools\Scripts\md5sum.py
    """
    instant_assert(isinstance(text, str), "Expecting string.")
    instant_assert(isinstance(filenames, (list,tuple)), "Expecting sequence.")
    
    m = hashlib.new("md5")
    if text:
        m.update(text)
    
    for filename in sorted(filenames): 
        instant_debug("Adding file '%s' to md5 sum." % filename)
        try:
            fp = open(filename, 'rb')
        except IOError, e:
            instant_error("Can't open file '%s': %s" % (filename, e))
        
        try:
            while 1:
                data = fp.read()
                if not data:
                    break
                m.update(data)
        except IOError, e:
            instant_error("I/O error reading '%s': %s" % (filename, e))
        
        fp.close() 
    
    return m.hexdigest().upper()


def _test():
    signature = "(Test signature)"
    files = ["signatures.py", "__init__.py"]
    print
    print "Signature:", repr(signature)
    print "MD5 sum:", compute_md5(signature, [])
    print
    print "files:", files
    print "MD5 sum:", compute_md5("", files)
    print

if __name__ == "__main__":
    _test()

