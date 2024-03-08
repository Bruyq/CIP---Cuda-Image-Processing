#include "kernel.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


/* KERNELS */
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


/* WRAPPERS */
__host__ void meanFilter(Image& dst, Image& src, int radius)
{
    /* Verify that sizes are the same */
    if (dst.getWidth() != src.getWidth() || dst.getHeight() != src.getHeight() || dst.getChannels() != src.getChannels() || radius > src.getWidth() / 2 || radius > src.getHeight() / 2)
    {
        std::cout << "Input and output images don't have the same dimensions or radius is too high" << std::endl;
        return;
    }
    else
    {
        /* Initialisation */
        size_t baseSize = src.getSize() * sizeof(unsigned char);
        int paddedWidth = src.getWidth() + 2 * radius;
        int paddedHeight = src.getHeight() + 2 * radius;
        size_t paddedSize = paddedWidth * paddedHeight * src.getChannels() * sizeof(unsigned char);
        unsigned char* d_padded1, *d_padded2, *d_src;

        /* Memory allocation on device */
        checkCudaErrors(cudaMalloc((void**)&d_src, baseSize));
        checkCudaErrors(cudaMalloc((void**)&d_padded1, paddedSize));
        checkCudaErrors(cudaMalloc((void**)&d_padded2, paddedSize));

        /* Copy to device memory */
        checkCudaErrors(cudaMemcpy(d_src, src.getData(), baseSize, cudaMemcpyHostToDevice));

        /* Extand base array to have the correct result on borders */
        replicate(d_padded1, d_src, src.getWidth(), src.getHeight(), radius, src.getChannels());

        /* Run device function */
        dim3 block_dim(32, 32);
        dim3 grid_dim((paddedWidth * src.getChannels() + block_dim.x - 1) / block_dim.x, (paddedHeight + block_dim.y - 1) / block_dim.y);
        clock_t timer = clock();
        meanFilterKernel << <grid_dim, block_dim >> > (d_padded2, d_padded1, paddedWidth, paddedHeight, radius, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        std::cout << "Duration of meanKernel : " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds" << std::endl;


        /* Crop the result to get relevant data */
        crop(d_src, d_padded2, radius, radius, src.getWidth(), src.getHeight(), paddedWidth, paddedHeight, src.getChannels());

        /* Retrieve result to host memory */
        checkCudaErrors(cudaMemcpy(dst.getData(), d_src, baseSize, cudaMemcpyDeviceToHost));

        /* Free memory */
        checkCudaErrors(cudaFree(d_src));
        checkCudaErrors(cudaFree(d_padded1));
        checkCudaErrors(cudaFree(d_padded2));
    }
}


__host__ void laplacianFilter(Image& dst, Image& src)
{
    /* Verify that sizes are the same */
    if (dst.getWidth() != src.getWidth() || dst.getHeight() != src.getHeight() || dst.getChannels() != src.getChannels())
    {
        std::cout << "Input and output images don't have the same dimensions or radius is too high" << std::endl;
        return;
    }
    else
    {
        /* Initialisation */
        size_t baseSize = src.getSize() * sizeof(unsigned char);
        int paddedWidth = src.getWidth() + 2;
        int paddedHeight = src.getHeight() + 2;
        size_t paddedSize = paddedWidth * paddedHeight * src.getChannels() * sizeof(unsigned char);
        unsigned char* d_padded1, * d_padded2, * d_src;

        /* Memory allocation on device */
        checkCudaErrors(cudaMalloc((void**)&d_src, baseSize));
        checkCudaErrors(cudaMalloc((void**)&d_padded1, paddedSize));
        checkCudaErrors(cudaMalloc((void**)&d_padded2, paddedSize));

        /* Copy to device memory */
        checkCudaErrors(cudaMemcpy(d_src, src.getData(), baseSize, cudaMemcpyHostToDevice));

        /* Extand base array to have the correct result on borders */
        replicate(d_padded1, d_src, src.getWidth(), src.getHeight(), 1, src.getChannels());

        /* Run device function */
        dim3 block_dim(32, 32);
        dim3 grid_dim((paddedWidth * src.getChannels() + block_dim.x - 1) / block_dim.x, (paddedHeight + block_dim.y - 1) / block_dim.y);
        clock_t timer = clock();
        laplacianFilterKernel << <grid_dim, block_dim >> > (d_padded2, d_padded1, paddedWidth, paddedHeight, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        std::cout << "Duration of laplacianKernel : " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds" << std::endl;


        /* Crop the result to get relevant data */
        crop(d_src, d_padded2, 1, 1, src.getWidth(), src.getHeight(), paddedWidth, paddedHeight, src.getChannels());

        /* Retrieve result to host memory */
        checkCudaErrors(cudaMemcpy(dst.getData(), d_src, baseSize, cudaMemcpyDeviceToHost));

        /* Free memory */
        checkCudaErrors(cudaFree(d_src));
        checkCudaErrors(cudaFree(d_padded1));
        checkCudaErrors(cudaFree(d_padded2));
    }
}


__host__ void generateRGB(Image& img)
{
    /* Initialisation */
    size_t arraySize = img.getSize() * sizeof(unsigned char);
    unsigned char* d_dest;
    dim3 block_dim(3, 1);
    dim3 grid_dim((img.getWidth() * img.getChannels() + block_dim.x - 1) / block_dim.x, (img.getHeight() + block_dim.y - 1) / block_dim.y);

    /* Memory allocation on device */
    checkCudaErrors(cudaMalloc((void**)&d_dest, arraySize));

    /* Run device function */
    generateRGBKernel << <grid_dim, block_dim >> > (d_dest, img.getWidth(), img.getHeight());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(img.getData(), d_dest, arraySize, cudaMemcpyDeviceToHost));

    /* Free memory */
    checkCudaErrors(cudaFree(d_dest));
}


__host__ void replicate(Image& dst, Image& src, int radius)
{
    /* Verify radius is below (img1_dims/2 + 1) and that img1 dims are superior to img2 dims*/
    if (radius > src.getWidth() - 1 || radius > src.getHeight() - 1 || src.getWidth() + 2 * radius != dst.getWidth() || src.getHeight() + 2 * radius != dst.getHeight())
    {
        std::cout <<
            "Case not handled : either the radius is too high for this image, either the dimensions of the images are invalid"
            << std::endl;
    }
    else
    {
        /* Initialisation */
        unsigned char* d_dest, * d_src;

        /* Memory allocation on device */
        checkCudaErrors(cudaMalloc((void**)&d_src, src.getSize() * sizeof(unsigned char)));
        checkCudaErrors(cudaMalloc((void**)&d_dest, dst.getSize() * sizeof(unsigned char)));

        /* Copy to device memory */
        checkCudaErrors(cudaMemcpy(d_src, src.getData(), src.getSize() * sizeof(unsigned char), cudaMemcpyHostToDevice));

        /* Call device function */
        replicate(d_dest, d_src, src.getWidth(), src.getHeight(), radius, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        /* Retrieve in host memory */
        checkCudaErrors(cudaMemcpy(dst.getData(), d_dest, dst.getSize() * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        /* Free memory */
        checkCudaErrors(cudaFree(d_src));
        checkCudaErrors(cudaFree(d_dest));
    }
}


__host__ void replicate(unsigned char* d_dest, unsigned char* d_src, int width, int height, int radius, int channels)
{
    /* Initialisation */
    clock_t timer;
    dim3 block_dim(32, 32);
    dim3 grid_dim(((width + 2 * radius) * channels + block_dim.x - 1) / block_dim.x, (height + 2 * radius + block_dim.y - 1) / block_dim.y);

    /* Run device function */
    timer = clock();
    replicateKernel << <grid_dim, block_dim >> > (d_dest, d_src, width, height, radius, channels);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Duration of replicateKernel : " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds" << std::endl; // 512x512 -> 1024x1024 in 5ms
}


__host__ void crop(Image& dst, Image& src, int posX, int posY, int width, int height)
{
    /* Verify size requirements */
    if (src.getWidth() <= width + posX || src.getHeight() <= height + posY || dst.getWidth() > width || dst.getHeight() > height)
    {
        std::cout <<
            "Can't crop : Invalid dimensions"
            << std::endl;
    }
    else
    {
        /* Initialisation */
        unsigned char* d_dest, * d_src;
        
        /* Memory allocation on device */
        checkCudaErrors(cudaMalloc((void**)&d_src, src.getSize() * sizeof(unsigned char)));
        checkCudaErrors(cudaMalloc((void**)&d_dest, dst.getSize() * sizeof(unsigned char)));

        /* Copy to device memory */
        checkCudaErrors(cudaMemcpy(d_src, src.getData(), src.getSize() * sizeof(unsigned char), cudaMemcpyHostToDevice));

        /* Call device function */
        crop(d_dest, d_src, posX, posY, dst.getWidth(), dst.getHeight(), src.getWidth(), src.getHeight(), dst.getChannels());

        /* Retrieve in host memory */
        checkCudaErrors(cudaMemcpy(dst.getData(), d_dest, dst.getSize() * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        /* Free memory */
        checkCudaErrors(cudaFree(d_src));
        checkCudaErrors(cudaFree(d_dest));
    }
}


__host__ void crop(unsigned char* d_dest, unsigned char* d_src, int posX, int posY, int width, int height, int widthInit, int heightInit, int channels)
{
    /* Initialisation */
    clock_t timer;
    dim3 block_dim(32, 32);
    dim3 grid_dim((width * channels + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);

    /* Run device function */
    timer = clock();
    cropKernel << <grid_dim, block_dim >> > (d_dest, d_src, posX, posY, width, height, widthInit, channels); // very fast - depends on the case ?
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Duration of cropKernel : " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds" << std::endl;
}


__host__ void guidedFilterSmoothing(Image& dst, Image& src, int radius)
{
    /* Verify that sizes are the same */
    if (dst.getWidth() != src.getWidth() || dst.getHeight() != src.getHeight() || dst.getChannels() != src.getChannels() || 2 * radius > src.getWidth() / 2 || 2 * radius > src.getHeight() / 2)
    {
        std::cout << "Input and output images don't have the same dimensions or radius is too high" << std::endl;
        return;
    }
    else
    {
        /* Initialisation */
        size_t baseSize = src.getSize() * sizeof(unsigned char);
        int paddedWidth = src.getWidth() + 4 * radius;
        int paddedHeight = src.getHeight() + 4 * radius;
        size_t paddedSizeUC = paddedWidth * paddedHeight * src.getChannels() * sizeof(unsigned char);
        size_t paddedSizeF = paddedWidth * paddedHeight * src.getChannels() * sizeof(float);
        unsigned char* d_padded1, * d_padded2, * d_src;
        float* d_ak, * d_bk;
        clock_t timer = clock();

        /* Memory allocation on device */
        checkCudaErrors(cudaMalloc((void**)&d_src, baseSize));
        checkCudaErrors(cudaMalloc((void**)&d_padded1, paddedSizeUC));
        checkCudaErrors(cudaMalloc((void**)&d_padded2, paddedSizeUC));
        checkCudaErrors(cudaMalloc((void**)&d_ak, paddedSizeF));
        checkCudaErrors(cudaMalloc((void**)&d_bk, paddedSizeF));

        /* Copy to device memory */
        checkCudaErrors(cudaMemcpy(d_src, src.getData(), baseSize, cudaMemcpyHostToDevice));

        /* Extand base array to have the correct result on borders */
        replicate(d_padded1, d_src, src.getWidth(), src.getHeight(), 2 * radius, src.getChannels());

        /* Run device function */
        dim3 block_dim(32, 32);
        dim3 grid_dim((paddedWidth * src.getChannels() + block_dim.x - 1) / block_dim.x, (paddedHeight + block_dim.y - 1) / block_dim.y);
        guidedFirstKernel << <grid_dim, block_dim >> > (d_ak, d_bk, d_padded1, paddedWidth, paddedHeight, radius, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        guidedSecondKernel << <grid_dim, block_dim >> > (d_padded2, d_ak, d_bk, d_padded1, paddedWidth, paddedHeight, radius, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        /* Crop the result to get relevant data */
        crop(d_src, d_padded2, 2 * radius, 2 * radius, src.getWidth(), src.getHeight(), paddedWidth, paddedHeight, src.getChannels());

        /* Retrieve result to host memory */
        checkCudaErrors(cudaMemcpy(dst.getData(), d_src, baseSize, cudaMemcpyDeviceToHost));
        std::cout << "Duration of guidedFilterSmoothing : " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds" << std::endl;

        /* Free memory */
        checkCudaErrors(cudaFree(d_src));
        checkCudaErrors(cudaFree(d_padded1));
        checkCudaErrors(cudaFree(d_padded2));
        checkCudaErrors(cudaFree(d_ak));
        checkCudaErrors(cudaFree(d_bk));
    }
}


__host__ void guidedFilterEnhancement(Image& dst, Image& src, int radius, float value)
{
    /* Verify that sizes are the same */
    if (dst.getWidth() != src.getWidth() || dst.getHeight() != src.getHeight() || dst.getChannels() != src.getChannels() || 2 * radius > src.getWidth() / 2 || 2 * radius > src.getHeight() / 2)
    {
        std::cout << "Input and output images don't have the same dimensions or radius is too high" << std::endl;
        return;
    }
    else
    {
        /* Initialisation */
        size_t baseSize = src.getSize() * sizeof(unsigned char);
        int paddedWidth = src.getWidth() + 4 * radius;
        int paddedHeight = src.getHeight() + 4 * radius;
        size_t paddedSizeUC = paddedWidth * paddedHeight * src.getChannels() * sizeof(unsigned char);
        size_t paddedSizeF = paddedWidth * paddedHeight * src.getChannels() * sizeof(float);
        unsigned char* d_padded1, * d_padded2, * d_src, * d_src2, * d_dst;
        float* d_ak, * d_bk;
        clock_t timer = clock();

        /* Memory allocation on device */
        checkCudaErrors(cudaMalloc((void**)&d_src, baseSize));
        checkCudaErrors(cudaMalloc((void**)&d_src2, baseSize));
        checkCudaErrors(cudaMalloc((void**)&d_dst, baseSize));
        checkCudaErrors(cudaMalloc((void**)&d_padded1, paddedSizeUC));
        checkCudaErrors(cudaMalloc((void**)&d_padded2, paddedSizeUC));
        checkCudaErrors(cudaMalloc((void**)&d_ak, paddedSizeF));
        checkCudaErrors(cudaMalloc((void**)&d_bk, paddedSizeF));

        /* Copy to device memory */
        checkCudaErrors(cudaMemcpy(d_src, src.getData(), baseSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_src2, src.getData(), baseSize, cudaMemcpyHostToDevice));

        /* Extand base array to have the correct result on borders */
        replicate(d_padded1, d_src, src.getWidth(), src.getHeight(), 2 * radius, src.getChannels());

        /* Run device function */
        dim3 block_dim(32, 32);
        dim3 grid_dim((paddedWidth * src.getChannels() + block_dim.x - 1) / block_dim.x, (paddedHeight + block_dim.y - 1) / block_dim.y);
        guidedFirstKernel << <grid_dim, block_dim >> > (d_ak, d_bk, d_padded1, paddedWidth, paddedHeight, radius, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        guidedSecondKernel << <grid_dim, block_dim >> > (d_padded2, d_ak, d_bk, d_padded1, paddedWidth, paddedHeight, radius, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        /* Crop the result to get relevant data */
        crop(d_src, d_padded2, 2 * radius, 2 * radius, src.getWidth(), src.getHeight(), paddedWidth, paddedHeight, src.getChannels());

        grid_dim.x = (src.getWidth() * src.getChannels() + block_dim.x - 1) / block_dim.x;
        grid_dim.y = (src.getHeight() + block_dim.y - 1) / block_dim.y;
        unsharpMaskingKernel << <grid_dim, block_dim >> > (d_dst, d_src2, d_src, dst.getWidth(), dst.getHeight(), value, dst.getChannels());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        /* Retrieve result to host memory */
        checkCudaErrors(cudaMemcpy(dst.getData(), d_dst, baseSize, cudaMemcpyDeviceToHost));
        std::cout << "Duration of guidedFilterEnhancement : " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds" << std::endl;

        /* Free memory */
        checkCudaErrors(cudaFree(d_src));
        checkCudaErrors(cudaFree(d_src2));
        checkCudaErrors(cudaFree(d_dst));
        checkCudaErrors(cudaFree(d_padded1));
        checkCudaErrors(cudaFree(d_padded2));
        checkCudaErrors(cudaFree(d_ak));
        checkCudaErrors(cudaFree(d_bk));
    }
}