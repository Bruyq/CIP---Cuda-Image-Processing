#include "helperfunc.h"

// File selection window function
// Opens a window in which one can choose the file to open
TCHAR* getFilename()
{
    LPSTR filebuff = new char[256];
    OPENFILENAME open = { 0 };
    open.lStructSize = sizeof(OPENFILENAME);
    open.hwndOwner = NULL;
    open.lpstrFilter = "Image Files(.jpg|.png|.bmp|.jpeg)\0*.jpg;*.png;*.bmp;*.jpeg\0\0";
    open.lpstrCustomFilter = NULL;
    open.lpstrFile = filebuff;
    open.lpstrFile[0] = '\0';
    open.nMaxFile = 256;
    open.nFilterIndex = 1;
    open.lpstrInitialDir = NULL;
    open.lpstrTitle = "Select An Image File\0";
    open.nMaxFileTitle = strlen("Select an image file\0");
    open.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_EXPLORER;

    if (GetOpenFileName(&open) == TRUE)
    {
        return open.lpstrFile;
    }
    else
    {
        std::cout << "Impossible d'ouvrir le fichier" << std::endl;
        return NULL;
    }
}


// Operation selection on image
Image selectOperation(Image img)
{
    int input = -1;
    int radius, length, width, height, posX, posY;
    float factor;

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
            std::endl;
        std::cin >> input;
    }
    switch (input)
    {
    case 0:
        radius = 0;
        while (radius < 1)
        {
            std::cout << "Please select Mean filter radius (where radius > 0) :" << std::endl;
            std::cin >> radius;
        }
        meanFilter(img, img, radius);
        break;
    case 1:
        laplacianFilter(img, img);
        break;
    case 2:
        radius = 0;
        while (radius < 1)
        {
            std::cout << "Please select guided filter radius (where radius > 0) :" << std::endl;
            std::cin >> radius;
        }
        guidedFilterSmoothing(img, img, radius);
        break;
    case 3:
        radius = 0;
        while (radius < 1)
        {
            std::cout << "Please select guided filter radius (where radius > 0) :" << std::endl;
            std::cin >> radius;
        }
        factor = 0;
        while (factor <= 0)
        {
            std::cout << "Please select guided filter enhancement factor (where factor > 0) :" << std::endl;
            std::cin >> factor;
        }
        guidedFilterEnhancement(img, img, radius, factor);
        break;
    case 4:
    {
        length = 0;
        while (length < 1)
        {
            std::cout << "Please select replicate padding length (where length > 0) :" << std::endl;
            std::cin >> length;
        }
        Image replicat(img.getWidth() + 2 * length, img.getHeight() + 2 * length, img.getChannels());
        replicate(replicat, img, length);
        break;
    }
    case 5:
    {
        std::cout << "Image width : " << img.getWidth() << "\t Image height : " << img.getHeight() << std::endl;
        width = -1;
        while (width < 0 || width > img.getWidth())
        {
            std::cout << "Please select cropped image width (where width >= 0 and less than original image width) :" << std::endl;
            std::cin >> width;
        }
        height = -1;
        while (height < 0 || height > img.getHeight())
        {
            std::cout << "Please select cropped image height (where height >= 0 and less than original image height) :" << std::endl;
            std::cin >> height;
        }
        posX = -1;
        while (posX < 0 || posX > img.getWidth() - width)
        {
            std::cout << "Please select cropping point x coordinate (where x >= 0 and less than original image width - cropped width) :" << std::endl;
            std::cin >> posX;
        }
        posY = -1;
        while (posY < 0 || posY > img.getHeight() - height)
        {
            std::cout << "Please select cropping point y coordinate (where y >= 0 and less than original image height - cropped height) :" << std::endl;
            std::cin >> posY;
        }
        Image cropped(width, height, img.getChannels());
        crop(cropped, img, posX, posY, width, height);
        break;
    }
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
        binarize(img, img, target_channel, threshold);
        break;
    }
    case 7:
    {

    }
    default:

    }
    return img;
}