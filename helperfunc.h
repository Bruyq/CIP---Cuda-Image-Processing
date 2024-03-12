#ifndef HELPERFUNC_H
#define HELPERFUNC_H

#include <windows.h>
#include <shobjidl.h> 
#include <ShlObj_core.h>
#include <stdio.h>
#include <iostream>
#include "wrapper.cuh"

void getFilename(char* pathbuffer, char* msg = "Select a file");

void getFolder(char* pathbuffer, char* msg = "Select a folder");

Image selectOperation(Image img);

#endif HELPERFUNC_H