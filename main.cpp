#include "helperfunc.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "wrapper.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Nvidia geForce gtx970 4GB : compute capability is 5.2
// Maximum number of threads per block : 1024
// Maximum x or y dimensionality of a block : 1024
// Maximum z-dimension of a block : 64
// Maximum number of resident grids per device (Concurrent Kernel Execution) : 32
// REMEMBER : host function (cpu code) keeps working while GPU works. So if a kernel is waiting for GPU ressource, CPU code might be executed in the meantime.
int main()
{
    // File reading routine
    char* saveDir = "D:/TRAVAIL/Post/results/";
    char* name = getFilename();
    Image img(name);
    int radius = 7;
    Image res(img.getWidth(), img.getHeight(), img.getChannels());
    

    // Process image
    //guidedFilterSmoothing(res, img, radius);
    guidedFilterEnhancement(res, img, radius, 1.f);
    //replicate(res, img, radius);
    //laplacianFilter(res, img);

    // File saving routine
    char dest[50];
    strcpy(dest, saveDir);
    strcat(dest, "res.png");
    res.save(dest);

    return 0;
}