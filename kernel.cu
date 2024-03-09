#include "kernel.cuh"

// Applies mean filter to the data
// Parameters : 
// - "width", "height", "channels" for image dimensions
// - "radius" for mean filter radius
// - "dest" stands for destination data and "src" for source data
__global__ void meanFilterKernel(unsigned char* dest, const unsigned char* src, int width, int height, int radius, int channels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width * channels && y < height)
    {
        // if RGB image : One line has RGB values for each pixels, so width is in fact width * 3
        int index = (y * width * channels + x);
        float sum = 0.0f;

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) 
            {
                // Following x axis we need to multiply the offset by the number of channels to avoid comparison between two distinct channels
                int px = x + i * channels;
                int py = y + j;

                if (px >= 0 && px < width * channels && py >= 0 && py < height)
                {
                    sum += (float)src[py * width * channels + px];
                }
            }
        }
        dest[index] = (unsigned char)(sum / ((2 * radius + 1) * (2 * radius + 1)));
    }
}


// Applies unsharp masking to the data (performs detail enhancement)
// Parameters : 
// - "width", "height", "channels" for image dimensions
// - "factor" for detail amplification factor
// - "dest" stands for destination data, "src" for source data and "smoothed" is the smoothed version of src data
__global__ void unsharpMaskingKernel(unsigned char* dest, const unsigned char* src, const unsigned char* smoothed, int width, int height, float factor, int channels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width * channels && y < height)
    {
        // if RGB image : One line has RGB values for each pixels, so width is in fact width * 3
        int index = (y * width * channels + x);
        float val = src[index];
        float res = val + factor * (val - smoothed[index]);
        
        dest[index] = (unsigned char)(res > 255 ? 255 : res);
    }
}


// Applies laplacian filter to the data (useful for edge detection)
// Parameters : 
// - "width", "height", "channels" for image dimensions
// - "dest" stands for destination data and "src" for source data
__global__ void laplacianFilterKernel(unsigned char* dest, const unsigned char* src, int width, int height, int channels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float kernel[3][3] = { {0, -1, 0}, {-1, 4, -1}, {0, -1, 0} };
    //float kernel[3][3] = { {-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1} };

    if (x < width * channels && y < height)
    {
        // if RGB image : One line has RGB values for each pixels, so width is in fact width * 3
        int index = (y * width * channels + x);
        float sum = 0.0f;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++)
            {
                // Following x axis we need to multiply the offset by the number of channels to avoid comparison between two distinct channels
                int px = x + i * channels;
                int py = y + j;

                if (px >= 0 && px < width * channels && py >= 0 && py < height)
                {
                    sum += (float)src[py * width * channels + px] * kernel[i + 1][j + 1];
                }
            }
        }
        dest[index] = (unsigned char)(sum);
    }
}


// Generates test image data displaying color gradients
// Parameters : 
// - "width", "height", "channels" for image dimensions
// - "dest" stands for destination data and "src" for source data
__global__ void generateRGBKernel(unsigned char* dest, int width, int height)
{
    // Block dim must be (3, 1) so we can know the channel we are working on
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int factor = 3; // 3 RGB values
    if (x < width * factor && y < height)
    {
        // RGB image : One line has RGB values for each pixels, so width is in fact width * 3
        int index = (y * width * factor + x);
        if (threadIdx.x == 2)
        {
            // Blue channel is null
            dest[index] = (unsigned char)(0);
        }
        else
        {
            // Red channel increments following X direction, green channel increments following y direction
            dest[index] = (unsigned char)((threadIdx.x == 0) ? blockIdx.x : blockIdx.y);
        }
    }
}


// Pads image data with "replicate" option 
// Example : 
// - (10, 0, 23), (50, 60, 41), (23, 23, 23) | (23, 23, 23), (50, 60, 41), (10, 0, 23) where "|" is initial image border
// Parameters :
// - "width", "height", "channels" for image dimensions
// - "radius" is equal to the padding length on borders
// - "dest" stands for destination data and "src" for source data
__global__ void replicateKernel(unsigned char* dest, const unsigned char* src, int width, int height, int radius, int channels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = (y * (width + 2 * radius) * channels + x);

    // If image as 3 channels, data will be like : RGBRGBRGB... and so on, so we adapt the algorithm for 3 cases
    if (y < radius && x < radius * channels)                                                                                    // Top-left corner
    {
        if ((x - 2) % 3 == 0 && channels == 3)
		{
			// When it is supposed to be blue
			dest[index] = src[radius * channels - x + (radius - y) * width * channels + 1];
		}
		else if (x % 3 == 0 && channels == 3)
		{
			// When it is supposed to be red
			dest[index] = src[radius * channels - x + (radius - y) * width * channels - 3];
		}
		else
		{
			// When it is supposed to be green (no modification on the behavior, same as if it was a 1 channel image)
			dest[index] = src[radius * channels - x + (radius - y) * width * channels - 1];
		}
    }
	else if (y < radius && x < (width + radius) * channels)                                                                     // Upper Border
	{
		dest[index] = src[(radius - y) * width * channels + x - radius * channels];
	}
	else if (y < radius && x < (width + 2 * radius) * channels)                                                                 // Top-right corner
	{
		if ((x - 2) % 3 == 0 && channels == 3)
		{
			// When it is supposed to be blue
			dest[index] = src[((radius - y + 1) * width + (width + radius)) * channels - x + 1];
		}
		else if (x % 3 == 0 && channels == 3)
		{
			// When it is supposed to be red
			dest[index] = src[((radius - y + 1) * width + (width + radius)) * channels - x - 3];
		}
		else
		{
			// When it is supposed to be green (no modification on the behavior, same as if it was a 1 channel image)
			dest[index] = src[((radius - y + 1) * width + (width + radius)) * channels - x - 1];
		}
	}
    else if (y < height + radius && x < radius * channels)                                                                      // Left border
    {
        if ((x - 2) % 3 == 0 && channels == 3)
        {
            // When it is supposed to be blue
            dest[index] = src[radius * channels - x + (y - radius) * width * channels + 1];
        }
        else if (x % 3 == 0 && channels == 3)
        {
            // When it is supposed to be red
            dest[index] = src[radius * channels - x + (y - radius) * width * channels - 3];
        }
        else
        {
            // When it is supposed to be green (no modification on the behavior, same as if it was a 1 channel image)
            dest[index] = src[radius * channels - x + (y - radius) * width * channels - 1];
        }
    }
    else if (y < height + radius && x < (width + radius) * channels)                                                            // Middle content
    {
        dest[index] = src[x - radius * channels + (y - radius) * width * channels];
    }
    else if (y < height + radius && x < (width + 2 * radius) * channels)                                                        // Right border
    {
        if ((x - 2) % 3 == 0 && channels == 3)
        {
            // When it is supposed to be blue
            dest[index] = src[((y - radius + 1) * width + (width + radius)) * channels - x + 1];
        }
        else if (x % 3 == 0 && channels == 3)
        {
            // When it is supposed to be red
            dest[index] = src[((y - radius + 1) * width + (width + radius)) * channels - x - 3];
        }
        else
        {
            // When it is supposed to be green (no modification on the behavior, same as if it was a 1 channel image)
            dest[index] = src[((y - radius + 1) * width + (width + radius)) * channels - x - 1];
        }
    }
    else if (y < 2 * radius + height && x < radius * channels)                                                                  // Bot-left corner
    {
		if ((x - 2) % 3 == 0 && channels == 3)
		{
			// When it is supposed to be blue
			dest[index] = src[(radius + 2 * height - y - 1) * width * channels + radius * channels - x + 1];
		}
		else if (x % 3 == 0 && channels == 3)
		{
			// When it is supposed to be red
			dest[index] = src[(radius + 2 * height - y - 1) * width * channels + radius * channels - x - 3];
		}
		else
		{
			// When it is supposed to be green (no modification on the behavior, same as if it was a 1 channel image)
			dest[index] = src[(radius + 2 * height - y - 1) * width * channels + radius * channels - x - 1];
		}
    }
    else if (y < 2 * radius + height && x < (width + radius) * channels)                                                        // Bottom border
    {
        dest[index] = src[(radius + 2 * height - y - 1) * width * channels + x - radius * channels];
    }
    else if (y < 2 * radius + height && x < (width + 2 * radius) * channels)                                                    // Bot-right corner
    {
        if ((x - 2) % 3 == 0 && channels == 3)
        {
            // When it is supposed to be blue
            dest[index] = src[((radius + 2 * height - y - 1) * width + (width + radius)) * channels - x + 1];
        }
        else if (x % 3 == 0 && channels == 3)
        {
            // When it is supposed to be red
            dest[index] = src[((radius + 2 * height - y - 1) * width + (width + radius)) * channels - x - 3];
        }
        else
        {
            // When it is supposed to be green (no modification on the behavior, same as if it was a 1 channel image)
            dest[index] = src[((radius + 2 * height - y - 1) * width + (width + radius)) * channels - x - 1];
        }
    }
}


// Crops image data
// Parameters :
// - "width", "height", "channels" for destination data dimensions
// - "widhtInit" for source data width and "(posX, posY)" is upper-left starting point for cropping in source image data
// - "dest" stands for destination data and "src" for source data
__global__ void cropKernel(unsigned char* dest, const unsigned char* src, int posX, int posY, int width, int height, int widthInit, int channels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = (y * width * channels + x);

    if (x < width * channels && y < height)
    {
        dest[index] = src[((y + posY) * widthInit  + posX) * channels + x];
    }
}


// Applies the first step for guided filter smoothing technique
// Parameters :
// - "width", "height", "channels" for destination data dimensions
// - "radius" gives the kernel radius for local mean and local variance computation
// - "ak" and "bk" are kernel data outputs and "src" for source data
__global__ void guidedFirstKernel(float* ak, float* bk, const unsigned char* src, int width, int height, int radius, int channels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width * channels && y < height)
    {
        // if RGB image : One line has RGB values for each pixels, so width is in fact width * 3
        int index = (y * width * channels + x);
        float mean = 0.0f;
        float w = (2 * radius + 1) * (2 * radius + 1);
        float var = 0.0f;
        float temp;

        for (int i = -radius; i <= radius; i++) 
        {
            for (int j = -radius; j <= radius; j++)
            {
                // Following x axis we need to multiply the offset by the number of channels to avoid comparison between two distinct channels
                int px = x + i * channels;
                int py = y + j;

                if (px >= 0 && px < width * channels && py >= 0 && py < height)
                {
                    temp = float(src[py * width * channels + px]) / 255;
                    mean += temp;
                    var += temp * temp;
                }
            }
        }
        mean = mean / w;
        var = var / w - mean * mean;
        ak[index] = ((var) / (var + 0.01f));
        bk[index] = ((1 - ak[index]) * mean);
    }
}


// Applies the second step for guided filter smoothing technique
// Parameters :
// - "width", "height", "channels" for destination data dimensions
// - "radius" gives the kernel radius for the local average computation
// - "dest" stands for destination data, "ak" and "bk" are data inputs and "src" stands for source data
__global__ void guidedSecondKernel(unsigned char* dest, const float* ak, const float* bk, const unsigned char* src, int width, int height, int radius, int channels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width * channels && y < height)
    {
        // if RGB image : One line has RGB values for each pixels, so width is in fact width * 3
        int index = (y * width * channels + x);
        float a = 0.0f;
        float w = (2 * radius + 1) * (2 * radius + 1);
        float b = 0.0f;

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++)
            {
                // Following x axis we need to multiply the offset by the numbre of channels to avoid comparison between two distinct channels
                int px = x + i * channels;
                int py = y + j;

                if (px >= 0 && px < width * channels && py >= 0 && py < height)
                {
                    a += ak[py * width * channels + px];
                    b += bk[py * width * channels + px];
                }
            }
        }
        a = a / w;
        b = b / w;
        dest[index] = (unsigned char)(a * float(src[index]) + b * 255);
    }
}


// Thresholds and binarize image data based on a specific channel.
// Parameters :
// - "width", "height", "channels" for destination data dimensions
// - "target_channel" is an integer that selects the channel to work on : 0 for red channel, 2 for blue, any other value for green
// - "threshold" selects the threshold value. Values higher or equal will be white (255), other values will be black (0). "treshold" should be in [0, 255]
// - "dest" stands for destination data and "src" for source data
__global__ void binarizeKernel(unsigned char* dest, unsigned char* src, int width, int height, int target_channel, int threshold, int channels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width * channels && y < height)
    {
        int index = (y * width + x) * channels;
 
        if (target_channel == 0 && channels == 3)                        // Thread working on red channel
        {
            if (src[index] >= threshold)
            {
                dest[index] = 255;
                dest[index + 1] = 255;
                dest[index + 2] = 255;
            }
            else
            {
                dest[index] = 0;
                dest[index + 1] = 0;
                dest[index + 2] = 0;
            }
        }
        else if (target_channel == 2 && channels == 3)             // Thread working on blue channel
        {
            if (src[index + 2] >= threshold)
            {
                dest[index] = 255;
                dest[index + 1] = 255;
                dest[index + 2] = 255;
            }
            else
            {
                dest[index] = 0;
                dest[index + 1] = 0;
                dest[index + 2] = 0;
            }
        }
        else if (channels == 3)                                    // Thread working on green channel
        {
            if (src[index + 1] >= threshold) 
            {
                dest[index] = 255;
                dest[index + 1] = 255;
                dest[index + 2] = 255;
            }
            else
            {
                dest[index] = 0;
                dest[index + 1] = 0;
                dest[index + 2] = 0;
            }
        }
        else                                                        // When working with 1 channel image
        {
            if (src[index] >= threshold)
            {
                dest[index] = 255;
            }
            else
            {
                dest[index] = 0;
            }
        }
    }
}

// TODO
__global__ void computeHistogramKernel(unsigned int* hist, unsigned char* src, int width, int height, int channels, int nbins)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width  && y < height)
    {
        int index = (y * width + x);
        if (channels == 1)
        {

        }
    }
}