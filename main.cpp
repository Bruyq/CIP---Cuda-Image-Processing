#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "helperfunc.h"


// Device is Nvidia geForce gtx970 4GB : compute capability is 5.2
// Maximum number of threads per block : 1024
// Maximum x or y dimensionality of a block : 1024
// Maximum z-dimension of a block : 64
// Maximum number of resident grids per device (Concurrent Kernel Execution) : 32
// REMEMBER : host function (cpu code) keeps working while GPU works. So if a kernel is waiting for GPU ressource, CPU code might be executed in the meantime.
int main()
{
    // Welcome message
    std::cout << "Hello World !\n" << "Welcome to this demo of some of image processing techniques I implemented with CUDA\n" <<
        "Device used while coding this was an Nvidia geForce gtx970 4GB with CUDA compute capability of 5.2\n" << 
        "Operating system used is windows and this demo makes calls to the win32 API\n" << std::endl;

    // Initialization
    bool flag = true;
    std::string result_name = "a";
    //char* saveDir = "D:/TRAVAIL/Post/results/";
    char* saveDir = getFolder("Select a folder where result images will be saved\0");

    while (flag)
    {
        // File reading routine
        std::cout << "Please choose an image to work on :" << std::endl;
        char* name = getFilename();
        Image img(name);
        Image res(img.getWidth(), img.getHeight(), img.getChannels());

        // Process image with available functions in wrapper.cuh
        res = selectOperation(img);

        // Output file name selection & image saving
        char dest[50];
        strcpy(dest, saveDir);
        std::cout << "Please enter the name of the file in which you want to save the result (wihout file extension) :" << std::endl;
        std::cin >> result_name;
        strcat(dest, result_name.c_str());
        strcat(dest, ".png");
        res.save(dest); // ISSUE WITH THIS PATH, CHAR* SEEM TO BE CHANGING WITHOUT ANY ASSIGNEMENT ???

        // Do we want to end the demo ?
        std::cout << "Do you want to keep using this demo ? [y/n]" << std::endl;
        std::string verif = "\0";
        while (verif != "y" && verif != "n")
        {
            std::cin >> verif;
        }
        if (verif == "n")
        {
            flag = false;
        }
    }
    
    return 0;
}