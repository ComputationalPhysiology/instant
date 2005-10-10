
LIBS = 
LDPATH = 

FLAGS = 

SWIG       = swig 
SWIGOPT    = -I.
INTERFACE  = test1_ext.i
TARGET     = test1_ext
INCLUDES   = 

SWIGMAKEFILE = $(SWIGSRC)/Examples/Makefile

python::
	$(MAKE) -f '$(SWIGMAKEFILE)' INTERFACE='$(INTERFACE)' \
	SWIG='$(SWIG)' SWIGOPT='$(SWIGOPT)'  \
        SRCS='' \
        CPPSRCS='' \
	INCLUDES='$(INCLUDES) -I.' \
        LIBS='$(LIBS) ' \
        CFLAGS='$(CFLAGS) $(FLAGS)' \
        TARGET='$(TARGET)' \
	python_cpp

clean::
	rm -f *_wrap* _test1_ext.so *.o $(OBJ_FILES)  *~
    