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