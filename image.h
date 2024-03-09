#ifndef IMAGE_H
#define IMAGE_H

#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"
#include <iostream>
#include <fstream>


// Class to handle image data
class Image
{
protected:
	unsigned char* m_data;
	int m_width;
	int m_height;
	int m_channels; // Need a case when m_channels = 1 (depth = 8, grayscale image) (doesn't work properly for now)

public:
	Image(char* filename);
	Image(int width = 1, int height = 1, int channels = 1);
	~Image();
	int getWidth();
	int getHeight();
	int getChannels();
	int getSize();
	void printData();
	void saveDataInPPM(const char* filename);
	unsigned char* getData();
	void save(const char* filename);
};
#endif