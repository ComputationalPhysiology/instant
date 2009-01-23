"""This module contains helper functions for configuration using pkg-config."""

import os
from output import get_status_output
import re

def get_swig_version(): 
    result, output = get_status_output("swig -version")
    if result != 0: 
        raise OSError("SWIG is not installed on the system.")
    pattern = "SWIG Version (.*)"
    r = re.search(pattern, output)
    return r.groups(0)[0]




def header_and_libs_from_pkgconfig(*packages, **kwargs):
    """This function returns list of include files, flags, libraries and library directories obtain from a pkgconfig file.
    
    The usage is: 
    (includes, flags, libraries, libdirs) = header_and_libs_from_pkgconfig(*list_of_packages)
    """
    returnLinkFlags = kwargs.get("returnLinkFlags", False)
    result, output = get_status_output("pkg-config --version ")
    if result != 0: 
        raise OSError("The pkg-config package is not installed on the system.")

    env = os.environ.copy()
    try:
        assert env["PKG_CONFIG_ALLOW_SYSTEM_CFLAGS"] == "0"
    except:
        env["PKG_CONFIG_ALLOW_SYSTEM_CFLAGS"] = "1"

    includes = []
    flags = []
    libs = []
    libdirs = []
    linkflags = []
    for pack in packages:
        result, output = get_status_output("pkg-config --exists %s " % pack, env=env)
        if result == 0: 
            tmp = get_status_output("pkg-config --cflags-only-I %s " % pack, env=env)[1].split()
            includes.extend(i[2:] for i in tmp)
            
            tmp = get_status_output("pkg-config --cflags-only-other %s " % pack, env=env)[1].split()
            flags.extend(tmp)
            
            tmp = get_status_output("pkg-config --libs-only-l  %s " % pack, env=env)[1].split()
            libs.extend(i[2:] for i in tmp)
            
            tmp = get_status_output("pkg-config --libs-only-L  %s " % pack, env=env)[1].split()
            libdirs.extend(i[2:] for i in tmp)

            tmp = get_status_output("pkg-config --libs-only-other  %s " % pack, env=env)[1].split()
            linkflags.extend(tmp)

        else: 
            raise OSError("The pkg-config file %s does not exist" % pack)

    if returnLinkFlags: return (includes,flags,libs, libdirs, linkflags) 
    return (includes,flags,libs, libdirs) 

