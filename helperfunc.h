#ifndef HELPERFUNC_H
#define HELPERFUNC_H

#include <windows.h>
#include <shobjidl.h> 
#include <stdio.h>
#include <iostream>
#include "wrapper.cuh"

TCHAR* getFilename();

Image selectOperation(Image img);

#endif HELPERFUNC_H