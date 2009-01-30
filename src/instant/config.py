"""This module contains helper functions for configuration using pkg-config."""

import os
from output import get_status_output
import re

def get_swig_version(): 
    """ Return the current swig version in a 'str'"""
    # Check for swig installation
    result, output = get_status_output("swig -version")
    if result != 0: 
        raise OSError("SWIG is not installed on the system.")
    pattern = "SWIG Version (.*)"
    r = re.search(pattern, output)
    return r.groups(0)[0]

def check_swig_version(version, same=False):
    """ Check the swig version

    Returns True if the version of the installed swig is equal or greater than the
    version passed to the function.

    If same is True, the function returns True if and only if the two versions
    are the same.
    
    Usage:
    if instant.check_swig_version('1.3.36'):
        print "Swig version is greater than or equal to 1.3.36"
    else:
        print "Swig version is lower than 1.3.36"
    """
    assert isinstance(version,str), "Provide the first version number as a 'str'"
    assert len(version.split("."))==3, "Provide the version number as three numbers seperated by '.'"

    installed_version = map(int, get_swig_version().split('.'))
    handed_version    = map(int, version.split('.'))
    
    # If same is True then just check that all numbers are equal
    if same:
        return all(i == h for i, h in zip(installed_version,handed_version))
    
    swig_enough = True
    for i, v in enumerate([v for v in installed_version]):
        if handed_version[i] < v:
            break
        elif handed_version[i] == v:
            continue
        else:
            swig_enough = False
        break
    
    return swig_enough

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

