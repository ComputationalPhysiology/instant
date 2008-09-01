

def inline(c_code, **args_dict):
    """
       This is a short wrapper around the create_extention function 
       in instant. 
       
       It creates an extension module given that
       the input is a valid C function. It is only possible
       to inline one C function each time. 

       Usage: 

       >>> from instant import inline
       >>> add_func = inline("double add(double a, double b){ return a+b; }")
       >>> print "The sum of 3 and 4.5 is ", add_func(3, 4.5)


    """
    args_dict["code"] = c_code
    if not args_dict.has_key("module"):
        args_dict["module"] = "inline_ext" 
    try: 
        func = c_code[:c_code.index('(')]
        ret, func_name = func.split()
        create_extension(**args_dict)
        exec("from inline_ext import %s as func_name"% func_name) 
        return func_name
    except: 
        create_extension(**args_dict)
        exec("import inline_ext as I") 
        return I 


def inline_with_numpy(c_code, **args_dict):
    """
       This is a short wrapper around the create_extention function 
       in instant. 
       
       It creates an extension module given that
       the input is a valid C function. It is only possible
       to inline one C function each time. The difference between
       this function and the inline function is that C-arrays can be used. 
       The following example illustrates that. 

       Usage: 

       >>> import numpy
       >>> import time
       >>> from instant import inline_with_numpy
       >>> c_code = \"\"\"
           double sum (int n1, double* array1){
               double tmp = 0.0; 
               for (int i=0; i<n1; i++) {  
                   tmp += array1[i]; 
               }
               return tmp; 
           }
           \"\"\"
       >>> sum_func = inline_with_numpy(c_code,  arrays = [['n1', 'array1']])
       >>> a = numpy.arange(10000000); a = numpy.sin(a)
       >>> sum_func(a)
    """

    func = c_code[:c_code.index('(')]
    ret, func_name = func.split()
    import numpy
    args_dict["code"] = c_code 
    args_dict["module"] = "inline_ext_numpy" 
    if args_dict.has_key("system_headers"):  
        args_dict["system_headers"].append ("arrayobject.h")
    else: 
        args_dict["system_headers"] = ["arrayobject.h"]

    if args_dict.has_key("include_dirs"): 
        args_dict["include_dirs"].append("%s/numpy"% numpy.get_include())
    else: 
        args_dict["include_dirs"] = ["%s/numpy"% numpy.get_include()]

    if args_dict.has_key("init_code"):
        args_dict["init_code"] += "\nimport_array();\n"
    else: 
        args_dict["init_code"] = "\nimport_array();\n"

    create_extension(**args_dict)
    exec("from inline_ext_numpy import %s as func_name"% func_name) 
    return func_name


def inline_with_numeric(c_code, **args_dict):
    """
       This is a short wrapper around the create_extention function 
       in instant. 
       
       It creates an extension module given that
       the input is a valid C function. It is only possible
       to inline one C function each time. The difference between
       this function and the inline function is that C-arrays can be used. 
       The following example illustrates that. 

       Usage: 

       >>> import numpy 
       >>> import time
       >>> from instant import inline_with_numeric
       >>> c_code = \"\"\"
           double sum (int n1, double* array1){
               double tmp = 0.0; 
               for (int i=0; i<n1; i++) {  
                   tmp += array1[i]; 
               }
               return tmp; 
           }
           \"\"\"
       >>> sum_func = inline_with_numeric(c_code,  arrays = [['n1', 'array1']])
       >>> a = numpy.arange(10000000); a = numpy.sin(a)
       >>> sum_func(a)
    """

    args_dict["code"] = c_code 
    args_dict["module"] = "inline_ext_numeric" 
    args_dict["system_headers"] = args_dict.get("system_headers", []) + ["arrayobject.h"]
    args_dict["include_dirs"] = args_dict.get("include_dirs", []) + \
        [[sys.prefix + "/include/python" + sys.version[:3] + "/Numeric", 
          sys.prefix + "/include" + "/Numeric"][sys.platform=='win32'],
          "/usr/local/include/python" + sys.version[:3] +  "/Numeric"]
    args_dict["init_code"] =  args_dict.get("init_code", "") + "\nimport_array();\n"

    try: 
        func = c_code[:c_code.index('(')]
        ret, func_name = func.split()
        create_extension(**args_dict)
        exec("from inline_ext_numeric import %s as func_name" % func_name)
        return func_name
    except: 
        create_extension(**args_dict)
        exec("import inline_ext_numeric as I") 
        return I  


def inline_with_numarray(c_code, **args_dict):
    """
       This is a short wrapper around the create_extention function 
       in instant. 
       
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

    func = c_code[:c_code.index('(')]
    ret, func_name = func.split()

    args_dict["code"] = c_code 
    args_dict["module"] = "inline_ext_numarray" 
    import numarray 
    inc_dir =  [numarray.numinclude.include_dir, 
                "/usr/local/include/python" + sys.version[:3] + "/numarray",
                "/usr/include/python" + sys.version[:3] + "/numarray" ] 

    if args_dict.has_key("system_headers"):  
        args_dict["system_headers"].append ("arrayobject.h")
    else: 
        args_dict["system_headers"] = ["arrayobject.h"]

    if args_dict.has_key("include_dirs"): 
        args_dict["include_dirs"].extend(inc_dir)
    else: 
        args_dict["include_dirs"] = inc_dir
    if args_dict.has_key("init_code"):
        args_dict["init_code"] += "\nimport_array();\n"
    else: 
        args_dict["init_code"] = "\nimport_array();\n"

    create_extension(**args_dict)

    exec("from inline_ext_numarray import %s as func_name"% func_name) 
    return func_name


