"""This module contains helper functions for configuration using pkg-config."""

from output import get_status_output

def header_and_libs_from_pkgconfig(*packages):
    """This function returns list of include files, flags, libraries and library directories obtain from a pkgconfig file.
    
    The usage is: 
    (includes, flags, libraries, libdirs) = header_and_libs_from_pkgconfig(*list_of_packages)
    """
    result, output = get_status_output("pkg-config --version ")
    if result != 0: 
        raise OSError("The pkg-config package is not installed on the system.")

    includes = []
    flags = []
    libs = []
    libdirs = []
    for pack in packages:
        result, output = get_status_output("pkg-config --exists %s " % pack)
        if result == 0: 
            tmp = get_status_output("pkg-config --cflags-only-I %s " % pack)[1].split()
            includes.extend(i[2:] for i in tmp)
            
            tmp = get_status_output("pkg-config --cflags-only-other %s " % pack)[1].split()
            flags.extend(tmp)
            
            tmp = get_status_output("pkg-config --libs-only-l  %s " % pack)[1].split()
            libs.extend(i[2:] for i in tmp)
            
            tmp = get_status_output("pkg-config --libs-only-L  %s " % pack)[1].split()
            libdirs.extend(i[2:] for i in tmp)
        else: 
            raise OSError("The pkg-config file %s does not exist" % pack)

    return (includes,flags,libs, libdirs) 

