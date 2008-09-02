
import sys
from output import instant_assert, instant_warning, instant_error
from create_extension import create_extension


def get_func_name(c_code):
    # TODO: Something more robust? Regexp?
    try:
        func = c_code[:c_code.index('(')]
        ret, func_name = func.split()
    except:
        instant_error("Failed to extract function name from c_code.")
    return func_name


def inline(c_code, **kwargs):
    """This is a short wrapper around the create_extention function in instant. 
    
    It creates an extension module given that
    the input is a valid C function. It is only possible
    to inline one C function each time. 

    Usage: 

    >>> from instant import inline
    >>> add_func = inline("double add(double a, double b){ return a+b; }")
    >>> print "The sum of 3 and 4.5 is ", add_func(3, 4.5)
    """
    instant_assert("code" not in kwargs, "Cannot specify code twice.")
    kwargs["code"] = c_code
    func_name = get_func_name(c_code)
    extension = create_extension(**kwargs)
    if hasattr(extension, func_name):
        return getattr(extension, func_name)
    else:
        instant_warning("Didn't find function '%s', returning module." % func_name)
    return extension


def inline_with_numpy(c_code, **kwargs):
    '''This is a short wrapper around the create_extention function in instant. 
       
    It creates an extension module given that
    the input is a valid C function. It is only possible
    to inline one C function each time. The difference between
    this function and the inline function is that C-arrays can be used. 
    The following example illustrates that. 

    Usage: 

    >>> import numpy
    >>> import time
    >>> from instant import inline_with_numpy
    >>> c_code = """
        double sum (int n1, double* array1){
            double tmp = 0.0; 
            for (int i=0; i<n1; i++) {  
                tmp += array1[i]; 
            }
            return tmp; 
        }
        """
    >>> sum_func = inline_with_numpy(c_code,  arrays = [['n1', 'array1']])
    >>> a = numpy.arange(10000000); a = numpy.sin(a)
    >>> sum_func(a)
    '''
    import numpy
    instant_assert("code" not in kwargs, "Cannot specify code twice.")
    kwargs["code"] = c_code 
    kwargs["system_headers"] = kwargs.get("system_headers",[]) + ["arrayobject.h"]
    kwargs["include_dirs"] = kwargs.get("include_dirs",[]) + ["%s/numpy"% numpy.get_include()]
    kwargs["init_code"] = kwargs.get("init_code",[]) + ["\nimport_array();\n"]
    func_name = get_func_name(c_code)
    extension = create_extension(**kwargs)
    if hasattr(extension, func_name):
        return getattr(extension, func_name)
    else:
        instant_warning("Didn't find function '%s', returning module." % func_name)
    return extension


def inline_with_numeric(c_code, **kwargs):
    '''This is a short wrapper around the create_extention function in instant.
       
    It creates an extension module given that
    the input is a valid C function. It is only possible
    to inline one C function each time. The difference between
    this function and the inline function is that C-arrays can be used. 
    The following example illustrates that. 

    Usage: 

    >>> import numpy 
    >>> import time
    >>> from instant import inline_with_numeric
    >>> c_code = """
        double sum (int n1, double* array1){
            double tmp = 0.0; 
            for (int i=0; i<n1; i++) {  
                tmp += array1[i]; 
            }
            return tmp; 
        }
        """
    >>> sum_func = inline_with_numeric(c_code,  arrays = [['n1', 'array1']])
    >>> a = numpy.arange(10000000); a = numpy.sin(a)
    >>> sum_func(a)
    '''
    instant_assert("code" not in kwargs, "Cannot specify code twice.")
    kwargs["code"] = c_code 
    kwargs["system_headers"] = kwargs.get("system_headers", []) + ["arrayobject.h"]
    kwargs["include_dirs"] = kwargs.get("include_dirs", []) + \
        [[sys.prefix + "/include/python" + sys.version[:3] + "/Numeric", 
          sys.prefix + "/include" + "/Numeric"][sys.platform=='win32'],
          "/usr/local/include/python" + sys.version[:3] +  "/Numeric"]
    kwargs["init_code"] =  kwargs.get("init_code", "") + "\nimport_array();\n"
    
    extension = create_extension(**kwargs)
    func_name = get_func_name(c_code)
    if hasattr(extension, func_name):
        return getattr(extension, func_name)
    else:
        instant_warning("Didn't find function '%s', returning module." % func_name)
    return extension


def inline_with_numarray(c_code, **kwargs):
    """This is a short wrapper around the create_extention function in instant. 
       
    It creates an extension module given that
    the input is a valid C function. It is only possible
    to inline one C function each time. The difference between
    this function and the inline function is that C-arrays can be used. 
    The following example illustrates that. 

    Usage: 

    >>> import numarray 
    >>> import time
    >>> from instant import inline_with_numarray
    >>> c_code = \"\"\"
        double sum (int n1, double* array1){
            double tmp = 0.0; 
            for (int i=0; i<n1; i++) {  
                tmp += array1[i]; 
            }
            return tmp; 
        }
        \"\"\"
    >>> sum_func = inline_with_numarray(c_code,  arrays = [['n1', 'array1']])
    >>> a = numarray.arange(10000000); a = numarray.sin(a)
    >>> sum_func(a)
    """
    import numarray 

    instant_assert("code" not in kwargs, "Cannot specify code twice.")
    kwargs["code"] = c_code 

    inc_dirs = [numarray.numinclude.include_dir, 
                "/usr/local/include/python" + sys.version[:3] + "/numarray",
                "/usr/include/python" + sys.version[:3] + "/numarray" ] 
    kwargs["system_headers"] = kwargs.get("system_headers",[]) + ["arrayobject.h"]
    kwargs["include_dirs"] = kwargs.get("include_dirs",[]) + inc_dirs
    kwargs["init_code"] = kwargs.get("init_code",[]) + ["\nimport_array();\n"]

    func_name = get_func_name(c_code)
    extension = create_extension(**kwargs)
    if hasattr(extension, func_name):
        return getattr(extension, func_name)
    else:
        instant_warning("Didn't find function '%s', returning module." % func_name)
    return extension

