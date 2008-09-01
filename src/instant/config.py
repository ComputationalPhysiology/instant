
import commands

def header_and_libs_from_pkgconfig(*packages):
    """This function returns list of include files, flags, libraries and library directories obtain from a pkgconfig file.
    
    The usage is: 
    (includes, flags, libraries, libdirs) = header_and_libs_from_pkgconfig(*list_of_packages)
    """
    includes = []
    flags = []
    libs = []
    libdirs = []
    for pack in packages:
        result, output = commands.getstatusoutput("pkg-config --exists %s " % pack)
        if result == 0: 
            tmp = string.split(commands.getoutput("pkg-config --cflags-only-I %s " % pack))
            includes.extend(i[2:] for i in tmp)
            
            tmp = string.split(commands.getoutput("pkg-config --cflags-only-other %s " % pack))
            flags.extend(tmp)
            
            tmp = string.split(commands.getoutput("pkg-config --libs-only-l  %s " % pack))
            libs.extend(i[2:] for i in tmp)
            
            tmp = string.split(commands.getoutput("pkg-config --libs-only-L  %s " % pack))
            libdirs.extend(i[2:] for i in tmp)
        else: 
            raise OSError("The pkg-config file %s does not exist" % pack)

    return (includes,flags,libs, libdirs) 

