#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "image.h"
#include <iostream>
#include <time.h>

/* KERNELS */
__global__ void meanFilterKernel(unsigned char* dest, const unsigned char* src, int width, int height, int radius, int channels = 3);

__global__ void unsharpMaskingKernel(unsigned char* dest, const unsigned char* src, const unsigned char* smoothed,  int width, int height, float factor, int channels = 3);

__global__ void laplacianFilterKernel(unsigned char* dest, const unsigned char* src, int width, int height, int channels = 3);

__global__ void guidedFirstKernel(float* ak, float* bk,  const unsigned char* src, int width, int height, int radius, int channels = 3);

__global__ void guidedSecondKernel(unsigned char* dest, const float* ak, const float* bk,  const unsigned char* src, int width, int height, int radius, int channels = 3);

__global__ void generateRGBKernel(unsigned char* dest, int width, int height);

__global__ void replicateKernel(unsigned char* dest, const unsigned char* src, int width, int height, int radius, int channels = 3);

__global__ void cropKernel(unsigned char* dest, const unsigned char* src, int posX, int posY, int width, int height, int widthInit, int channels = 3);


/* WRAPPERS */
__host__ void meanFilter(Image& dst, Image& src, int radius);

__host__ void laplacianFilter(Image& dst, Image& src);

__host__ void guidedFilterSmoothing(Image& dst, Image& src, int radius);

__host__ void guidedFilterEnhancement(Image& dst, Image& src, int radius, float value);

__host__ void generateRGB(Image& img);

__host__ void replicate(Image& dst, Image& src, int radius);

__host__ void replicate(unsigned char* d_dest, unsigned char* d_src, int width, int height, int radius, int channels = 3);

__host__ void crop(Image& dst, Image& src, int posX, int posY, int width, int height);

__host__ void crop(unsigned char* d_dest, unsigned char* d_src, int posX, int posY, int width, int height, int widthInit, int heightInit, int channels = 3);
#endif // KERNEL_CUH