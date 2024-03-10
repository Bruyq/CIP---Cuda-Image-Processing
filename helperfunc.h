#ifndef HELPERFUNC_H
#define HELPERFUNC_H

#include <windows.h>
#include <shobjidl.h> 
#include <ShlObj_core.h>
#include <stdio.h>
#include <iostream>
#include "wrapper.cuh"

TCHAR* getFilename();

TCHAR* getFolder(char* msg);

Image selectOperation(Image img);

#endif HELPERFUNC_H