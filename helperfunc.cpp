#include "helperfunc.h"

// File selection window function
// Opens a window in which one can choose the file to open
void getFilename(char* pathbuffer, char* msg)
{
    // Initialize a OPENFILENAME object
    OPENFILENAME open = { 0 };
    open.lStructSize = sizeof(OPENFILENAME);
    open.hwndOwner = NULL;
    open.lpstrFilter = "Image Files(.jpg|.png|.bmp|.jpeg)\0*.jpg;*.png;*.bmp;*.jpeg\0\0";
    open.lpstrCustomFilter = NULL;
    open.lpstrFile = pathbuffer;
    open.lpstrFile[0] = '\0';
    open.nMaxFile = 256;
    open.nFilterIndex = 1;
    open.lpstrInitialDir = NULL;
    open.lpstrTitle = msg;
    open.nMaxFileTitle = strlen(msg);
    open.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_EXPLORER;

    // Open dialog window using the OPENFILENAME object, handle different cases
    if (GetOpenFileName(&open) == TRUE)
    {
    }
    else
    {
        std::cout << "Cannot open the file" << std::endl;
    }
}


// Folder selection window function
// Opens a window in which one can choose the folder to select
// Parameter :
// - "pathbuffer" char ptr to the memory space where to save the folder path
// - "msg" message to display in dialog window
void getFolder(char* pathbuffer, char* msg)
{
    // Initialize a BROWSEINFOA object
    TCHAR folder_name[MAX_PATH];
    BROWSEINFOA bInfo = { 0 };
    bInfo.hwndOwner = NULL;
    bInfo.pidlRoot = NULL;
    bInfo.pszDisplayName = folder_name;
    bInfo.ulFlags = 0;
    bInfo.lpszTitle = msg;

    // Open dialog window using the BROWSEINFOA object, get PIDL object and retrieve the path from this PIDL, handle different cases
    LPCITEMIDLIST pidl = SHBrowseForFolderA(&bInfo);
    while (pidl == NULL)
    {
        pidl = SHBrowseForFolderA(&bInfo);
    }
    SHGetPathFromIDList(pidl, pathbuffer);
}


// Operation selection on image
// Displays in standard output image processing algorithms to apply to the image "img"
// Parameter :
// - "img" image on which we are currently working
Image selectOperation(Image img)
{
    // Initialization
    int input = -1;

    // Accept only valid options, else repeat the question
    while (input < 0 || input > 7)
    {
        std::cout <<
            "Select an operation to perform on choosen image :\n" <<
            "0 - Mean filter\n" <<
            "1 - Laplacian filter\n" <<
            "2 - GuidedFilter smoothing\n" <<
            "3 - GuidedFilter detail enhancement\n" <<
            "4 - Replicate borders\n" <<
            "5 - Crop image\n" <<
            "6 - Apply threshold\n" <<
            "7 - Apply mask\n" <<
            "8 - Quit" <<
            std::endl;
        std::cin >> input;
    }

    // For each option, a specific scenario occurs
    switch (input)
    {
        // Mean filter
        case 0:
        {
            int radius = 0;
            while (radius < 1)
            {
                std::cout << "Please select Mean filter radius (where radius > 0) :" << std::endl;
                std::cin >> radius;
            }
            Image res(img.getWidth(), img.getHeight(), img.getChannels());
            meanFilter(res, img, radius);
            return res;
        }

        // Laplacian Filter
        case 1:
        {
            Image res(img.getWidth(), img.getHeight(), img.getChannels());
            laplacianFilter(res, img);
            return res;
        }

        // Guided filter edge-preserving smoothing
        case 2:
        {
            int radius = 0;
            while (radius < 1)
            {
                std::cout << "Please select guided filter radius (where radius > 0) :" << std::endl;
                std::cin >> radius;
            }
            Image res(img.getWidth(), img.getHeight(), img.getChannels());
            guidedFilterSmoothing(res, img, radius);
            return res;
        }

        // Guided filter detail enhancement (smoothing + unsharp masking)
        case 3:
        {
            int radius = 0;
            while (radius < 1)
            {
                std::cout << "Please select guided filter radius (where radius > 0) :" << std::endl;
                std::cin >> radius;
            }
            float factor = 0;
            while (factor <= 0)
            {
                std::cout << "Please select guided filter enhancement factor (where factor > 0) :" << std::endl;
                std::cin >> factor;
            }
            Image res(img.getWidth(), img.getHeight(), img.getChannels());
            guidedFilterEnhancement(res, img, radius, factor);
            return res;
        }

        // Padding with replicate option
        case 4:
        {
            int length = 0;
            while (length < 1)
            {
                std::cout << "Please select replicate padding length (where length > 0) :" << std::endl;
                std::cin >> length;
            }

            Image res(img.getWidth() + 2 * length, img.getHeight() + 2 * length, img.getChannels());
            replicate(res, img, length);
            return res;
        }

        // Image cropping
        case 5:
        {
            std::cout << "Image width : " << img.getWidth() << "\t Image height : " << img.getHeight() << std::endl;
            int width = -1;
            while (width < 0 || width > img.getWidth())
            {
                std::cout << "Please select cropped image width (where width >= 0 and less than original image width) :" << std::endl;
                std::cin >> width;
            }
            int height = -1;
            while (height < 0 || height > img.getHeight())
            {
                std::cout << "Please select cropped image height (where height >= 0 and less than original image height) :" << std::endl;
                std::cin >> height;
            }
            int posX = -1;
            while (posX < 0 || posX > img.getWidth() - width)
            {
                std::cout << "Please select cropping point x coordinate (where x >= 0 and less than original image width - cropped width) :" << std::endl;
                std::cin >> posX;
            }
            int posY = -1;
            while (posY < 0 || posY > img.getHeight() - height)
            {
                std::cout << "Please select cropping point y coordinate (where y >= 0 and less than original image height - cropped height) :" << std::endl;
                std::cin >> posY;
            }
            Image res(width, height, img.getChannels());
            crop(res, img, posX, posY, width, height);
            return res;
        }

        // Thresholding
        case 6:
        {
            int target_channel = -1;
            int threshold = -1;
            while (target_channel < 0)
            {
                std::cout << "Please select target channel (0 = red, 2 = blue, other > 0 = green) :" << std::endl;
                std::cin >> target_channel;
            }
            while (threshold < 0 || threshold > 255)
            {
                std::cout << "Please select threshold value (where threshold is between 0 and 255) :" << std::endl;
                std::cin >> threshold;
            }
            Image res(img.getWidth(), img.getHeight(), img.getChannels());
            binarize(res, img, target_channel, threshold);
            return res;
        }

        // Masking
        case 7:
        {
            char* maskname = new char[MAX_PATH];
            std::cout << "Please select mask image :" << std::endl;
            getFilename(maskname, "Select the mask for the masking operation\0");
            Image mask(maskname);
            Image res(img.getWidth(), img.getHeight(), img.getChannels());
            masking(res, img, mask);
            delete[] maskname;
            return res;
        }
        case 8:
        {
            return img;
        }
    }
}
