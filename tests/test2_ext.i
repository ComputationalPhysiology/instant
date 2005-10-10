
%module test2_ext

%{
   #include <arrayobject.h>

#include <iostream>


PyObject* add(PyObject* a_, PyObject* b_){
  /*
  various checks
  */ 
  PyArrayObject* a=(PyArrayObject*) a_;
  PyArrayObject* b=(PyArrayObject*) b_;

  int n = a->dimensions[0];

  int dims[1];
  dims[0] = n; 
  PyArrayObject* ret;
  ret = (PyArrayObject*) PyArray_FromDims(1, dims, PyArray_DOUBLE); 

  int i;
  double aj;
  double bj;
  double *retj; 
  for (i=0; i < n; i++) {
    retj = (double*)(ret->data+ret->strides[0]*i); 
    aj = *(double *)(a->data+ a->strides[0]*i);
    bj = *(double *)(b->data+ b->strides[0]*i);
    *retj = aj + bj; 
  }
return PyArray_Return(ret);
}


%}

%init%{
import_array();
%}



PyObject* add(PyObject* a_, PyObject* b_);
    