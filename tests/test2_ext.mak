
LIBS = 
LDPATH = 

FLAGS = 

SWIG       = swig 
SWIGOPT    = -I.
INTERFACE  = test2_ext.i
TARGET     = test2_ext
INCLUDES   = 

SWIGMAKEFILE = $(SWIGSRC)/Examples/Makefile

python::
	$(MAKE) -f '$(SWIGMAKEFILE)' INTERFACE='$(INTERFACE)' \
	SWIG='$(SWIG)' SWIGOPT='$(SWIGOPT)'  \
        SRCS='' \
        CPPSRCS='' \
	INCLUDES='$(INCLUDES) -I/usr/include/python2.4/Numeric' \
        LIBS='$(LIBS) ' \
        CFLAGS='$(CFLAGS) $(FLAGS)' \
        TARGET='$(TARGET)' \
	python_cpp

clean::
	rm -f *_wrap* _test2_ext.so *.o $(OBJ_FILES)  *~
    