
try:
    from dolfin import *
except:
    print("dolfin not installed...")
    exit()

import instant

cpp_code = """
void dabla(dolfin::Vector& a, dolfin::Vector& b, double c, double d) {
    for (unsigned int i=0; i < a.size(); i++) {
        b.setitem(i, d*a[i] + c); 
    }
}
"""


a = Vector(12)
a[:] = 3.4
b = Vector(12)

c = 1.3 
d = 2.4 


include_dirs, flags, libs, libdirs = instant.header_and_libs_from_pkgconfig("dolfin")

headers= ["dolfin.h"]

func = instant.inline(cpp_code, system_headers=headers, include_dirs=include_dirs, libraries = libs, library_dirs = libdirs)  

func(a, b, c, d)

print(b.array())


