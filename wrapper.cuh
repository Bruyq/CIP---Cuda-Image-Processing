#ifndef WRAPPER_CUH
#define WRAPPER_CUH

#include "kernel.cuh"

__host__ void meanFilter(Image& dst, Image& src, int radius);

__host__ void laplacianFilter(Image& dst, Image& src);

__host__ void guidedFilterSmoothing(Image& dst, Image& src, int radius);

__host__ void guidedFilterEnhancement(Image& dst, Image& src, int radius, float value);

__host__ void generateRGB(Image& img);

__host__ void replicate(Image& dst, Image& src, int radius);

__host__ void replicate(unsigned char* d_dest, unsigned char* d_src, int width, int height, int radius, int channels = 3);

__host__ void crop(Image& dst, Image& src, int posX, int posY, int width, int height);

__host__ void crop(unsigned char* d_dest, unsigned char* d_src, int posX, int posY, int width, int height, int widthInit, int heightInit, int channels = 3);

#endif // WRAPPER_CUH