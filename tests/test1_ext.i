
%module test1_ext

%{

#include <iostream>


int hello(){
  printf("Hello World!\n");
  return 2222;
}


%}

%init%{
  //Code for initialisation here
%}



int hello();
    