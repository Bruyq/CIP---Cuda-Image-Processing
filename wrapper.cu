#include "wrapper.cuh"

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


__host__ void meanFilter(Image& dst, Image& src, int radius)
{
    // Verify that sizes are the same
    if (dst.getWidth() != src.getWidth() || dst.getHeight() != src.getHeight() || dst.getChannels() != src.getChannels() || radius > src.getWidth() / 2 || radius > src.getHeight() / 2)
    {
        std::cout << "Input and output images don't have the same dimensions or radius is too high" << std::endl;
        return;
    }
    else
    {
        // Initialisation
        size_t baseSize = src.getSize() * sizeof(unsigned char);
        int paddedWidth = src.getWidth() + 2 * radius;
        int paddedHeight = src.getHeight() + 2 * radius;
        size_t paddedSize = paddedWidth * paddedHeight * src.getChannels() * sizeof(unsigned char);
        unsigned char* d_padded1, * d_padded2, * d_src;

        // Memory allocation on device
        checkCudaErrors(cudaMalloc((void**)&d_src, baseSize));
        checkCudaErrors(cudaMalloc((void**)&d_padded1, paddedSize));
        checkCudaErrors(cudaMalloc((void**)&d_padded2, paddedSize));

        // Copy to device memory
        checkCudaErrors(cudaMemcpy(d_src, src.getData(), baseSize, cudaMemcpyHostToDevice));

        // Extand base array to have the correct result on borders
        replicate(d_padded1, d_src, src.getWidth(), src.getHeight(), radius, src.getChannels());

        // Run device function
        dim3 block_dim(32, 32);
        dim3 grid_dim((paddedWidth * src.getChannels() + block_dim.x - 1) / block_dim.x, (paddedHeight + block_dim.y - 1) / block_dim.y);
        clock_t timer = clock();
        meanFilterKernel << <grid_dim, block_dim >> > (d_padded2, d_padded1, paddedWidth, paddedHeight, radius, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        std::cout << "Duration of meanKernel : " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds" << std::endl;


        // Crop the result to get relevant data
        crop(d_src, d_padded2, radius, radius, src.getWidth(), src.getHeight(), paddedWidth, paddedHeight, src.getChannels());

        // Retrieve result to host memory
        checkCudaErrors(cudaMemcpy(dst.getData(), d_src, baseSize, cudaMemcpyDeviceToHost));

        // Free memory
        checkCudaErrors(cudaFree(d_src));
        checkCudaErrors(cudaFree(d_padded1));
        checkCudaErrors(cudaFree(d_padded2));
    }
}


__host__ void laplacianFilter(Image& dst, Image& src)
{
    // Verify that sizes are the same
    if (dst.getWidth() != src.getWidth() || dst.getHeight() != src.getHeight() || dst.getChannels() != src.getChannels())
    {
        std::cout << "Input and output images don't have the same dimensions or radius is too high" << std::endl;
        return;
    }
    else
    {
        // Initialisation
        size_t baseSize = src.getSize() * sizeof(unsigned char);
        int paddedWidth = src.getWidth() + 2;
        int paddedHeight = src.getHeight() + 2;
        size_t paddedSize = paddedWidth * paddedHeight * src.getChannels() * sizeof(unsigned char);
        unsigned char* d_padded1, * d_padded2, * d_src;

        // Memory allocation on device
        checkCudaErrors(cudaMalloc((void**)&d_src, baseSize));
        checkCudaErrors(cudaMalloc((void**)&d_padded1, paddedSize));
        checkCudaErrors(cudaMalloc((void**)&d_padded2, paddedSize));

        // Copy to device memory
        checkCudaErrors(cudaMemcpy(d_src, src.getData(), baseSize, cudaMemcpyHostToDevice));

        // Extand base array to have the correct result on borders
        replicate(d_padded1, d_src, src.getWidth(), src.getHeight(), 1, src.getChannels());

        // Run device function
        dim3 block_dim(32, 32);
        dim3 grid_dim((paddedWidth * src.getChannels() + block_dim.x - 1) / block_dim.x, (paddedHeight + block_dim.y - 1) / block_dim.y);
        clock_t timer = clock();
        laplacianFilterKernel << <grid_dim, block_dim >> > (d_padded2, d_padded1, paddedWidth, paddedHeight, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        std::cout << "Duration of laplacianKernel : " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds" << std::endl;


        // Crop the result to get relevant data
        crop(d_src, d_padded2, 1, 1, src.getWidth(), src.getHeight(), paddedWidth, paddedHeight, src.getChannels());

        // Retrieve result to host memory
        checkCudaErrors(cudaMemcpy(dst.getData(), d_src, baseSize, cudaMemcpyDeviceToHost));

        // Free memory
        checkCudaErrors(cudaFree(d_src));
        checkCudaErrors(cudaFree(d_padded1));
        checkCudaErrors(cudaFree(d_padded2));
    }
}


__host__ void generateRGB(Image& img)
{
    // Initialisation
    size_t arraySize = img.getSize() * sizeof(unsigned char);
    unsigned char* d_dest;
    dim3 block_dim(3, 1);
    dim3 grid_dim((img.getWidth() * img.getChannels() + block_dim.x - 1) / block_dim.x, (img.getHeight() + block_dim.y - 1) / block_dim.y);

    // Memory allocation on device
    checkCudaErrors(cudaMalloc((void**)&d_dest, arraySize));

    // Run device function
    generateRGBKernel << <grid_dim, block_dim >> > (d_dest, img.getWidth(), img.getHeight());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(img.getData(), d_dest, arraySize, cudaMemcpyDeviceToHost));

    // Free memory
    checkCudaErrors(cudaFree(d_dest));
}


__host__ void replicate(Image& dst, Image& src, int radius)
{
    // Verify radius is below (img1_dims/2 + 1) and that img1 dims are superior to img2 dims
    if (radius > src.getWidth() - 1 || radius > src.getHeight() - 1 || src.getWidth() + 2 * radius != dst.getWidth() || src.getHeight() + 2 * radius != dst.getHeight())
    {
        std::cout <<
            "Case not handled : either the radius is too high for this image, either the dimensions of the images are invalid"
            << std::endl;
    }
    else
    {
        // Initialisation
        unsigned char* d_dest, * d_src;

        // Memory allocation on device
        checkCudaErrors(cudaMalloc((void**)&d_src, src.getSize() * sizeof(unsigned char)));
        checkCudaErrors(cudaMalloc((void**)&d_dest, dst.getSize() * sizeof(unsigned char)));

        // Copy to device memory
        checkCudaErrors(cudaMemcpy(d_src, src.getData(), src.getSize() * sizeof(unsigned char), cudaMemcpyHostToDevice));

        // Call device function
        replicate(d_dest, d_src, src.getWidth(), src.getHeight(), radius, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Retrieve in host memory
        checkCudaErrors(cudaMemcpy(dst.getData(), d_dest, dst.getSize() * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        // Free memory
        checkCudaErrors(cudaFree(d_src));
        checkCudaErrors(cudaFree(d_dest));
    }
}


__host__ void replicate(unsigned char* d_dest, unsigned char* d_src, int width, int height, int radius, int channels)
{
    // Initialisation
    clock_t timer;
    dim3 block_dim(32, 32);
    dim3 grid_dim(((width + 2 * radius) * channels + block_dim.x - 1) / block_dim.x, (height + 2 * radius + block_dim.y - 1) / block_dim.y);

    // Run device function
    timer = clock();
    replicateKernel << <grid_dim, block_dim >> > (d_dest, d_src, width, height, radius, channels);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Duration of replicateKernel : " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds" << std::endl; // 512x512 -> 1024x1024 in 5ms
}


__host__ void crop(Image& dst, Image& src, int posX, int posY, int width, int height)
{
    // Verify size requirements
    if (src.getWidth() <= width + posX || src.getHeight() <= height + posY || dst.getWidth() > width || dst.getHeight() > height)
    {
        std::cout <<
            "Can't crop : Invalid dimensions"
            << std::endl;
    }
    else
    {
        // Initialisation
        unsigned char* d_dest, * d_src;

        // Memory allocation on device
        checkCudaErrors(cudaMalloc((void**)&d_src, src.getSize() * sizeof(unsigned char)));
        checkCudaErrors(cudaMalloc((void**)&d_dest, dst.getSize() * sizeof(unsigned char)));

        // Copy to device memory
        checkCudaErrors(cudaMemcpy(d_src, src.getData(), src.getSize() * sizeof(unsigned char), cudaMemcpyHostToDevice));

        // Call device function
        crop(d_dest, d_src, posX, posY, dst.getWidth(), dst.getHeight(), src.getWidth(), src.getHeight(), dst.getChannels());

        // Retrieve in host memory
        checkCudaErrors(cudaMemcpy(dst.getData(), d_dest, dst.getSize() * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        // Free memory
        checkCudaErrors(cudaFree(d_src));
        checkCudaErrors(cudaFree(d_dest));
    }
}


__host__ void crop(unsigned char* d_dest, unsigned char* d_src, int posX, int posY, int width, int height, int widthInit, int heightInit, int channels)
{
    // Initialisation
    clock_t timer;
    dim3 block_dim(32, 32);
    dim3 grid_dim((width * channels + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);

    // Run device function
    timer = clock();
    cropKernel << <grid_dim, block_dim >> > (d_dest, d_src, posX, posY, width, height, widthInit, channels); // very fast - depends on the case ?
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Duration of cropKernel : " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds" << std::endl;
}


__host__ void guidedFilterSmoothing(Image& dst, Image& src, int radius)
{
    // Verify that sizes are the same
    if (dst.getWidth() != src.getWidth() || dst.getHeight() != src.getHeight() || dst.getChannels() != src.getChannels() || 2 * radius > src.getWidth() / 2 || 2 * radius > src.getHeight() / 2)
    {
        std::cout << "Input and output images don't have the same dimensions or radius is too high" << std::endl;
        return;
    }
    else
    {
        // Initialisation
        size_t baseSize = src.getSize() * sizeof(unsigned char);
        int paddedWidth = src.getWidth() + 4 * radius;
        int paddedHeight = src.getHeight() + 4 * radius;
        size_t paddedSizeUC = paddedWidth * paddedHeight * src.getChannels() * sizeof(unsigned char);
        size_t paddedSizeF = paddedWidth * paddedHeight * src.getChannels() * sizeof(float);
        unsigned char* d_padded1, * d_padded2, * d_src;
        float* d_ak, * d_bk;
        clock_t timer = clock();

        // Memory allocation on device
        checkCudaErrors(cudaMalloc((void**)&d_src, baseSize));
        checkCudaErrors(cudaMalloc((void**)&d_padded1, paddedSizeUC));
        checkCudaErrors(cudaMalloc((void**)&d_padded2, paddedSizeUC));
        checkCudaErrors(cudaMalloc((void**)&d_ak, paddedSizeF));
        checkCudaErrors(cudaMalloc((void**)&d_bk, paddedSizeF));

        // Copy to device memory
        checkCudaErrors(cudaMemcpy(d_src, src.getData(), baseSize, cudaMemcpyHostToDevice));

        // Extand base array to have the correct result on borders
        replicate(d_padded1, d_src, src.getWidth(), src.getHeight(), 2 * radius, src.getChannels());

        // Run device function
        dim3 block_dim(32, 32);
        dim3 grid_dim((paddedWidth * src.getChannels() + block_dim.x - 1) / block_dim.x, (paddedHeight + block_dim.y - 1) / block_dim.y);
        guidedFirstKernel << <grid_dim, block_dim >> > (d_ak, d_bk, d_padded1, paddedWidth, paddedHeight, radius, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        guidedSecondKernel << <grid_dim, block_dim >> > (d_padded2, d_ak, d_bk, d_padded1, paddedWidth, paddedHeight, radius, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Crop the result to get relevant data
        crop(d_src, d_padded2, 2 * radius, 2 * radius, src.getWidth(), src.getHeight(), paddedWidth, paddedHeight, src.getChannels());

        // Retrieve result to host memory
        checkCudaErrors(cudaMemcpy(dst.getData(), d_src, baseSize, cudaMemcpyDeviceToHost));
        std::cout << "Duration of guidedFilterSmoothing : " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds" << std::endl;

        // Free memory
        checkCudaErrors(cudaFree(d_src));
        checkCudaErrors(cudaFree(d_padded1));
        checkCudaErrors(cudaFree(d_padded2));
        checkCudaErrors(cudaFree(d_ak));
        checkCudaErrors(cudaFree(d_bk));
    }
}


__host__ void guidedFilterEnhancement(Image& dst, Image& src, int radius, float value)
{
    // Verify that sizes are the same
    if (dst.getWidth() != src.getWidth() || dst.getHeight() != src.getHeight() || dst.getChannels() != src.getChannels() || 2 * radius > src.getWidth() / 2 || 2 * radius > src.getHeight() / 2)
    {
        std::cout << "Input and output images don't have the same dimensions or radius is too high" << std::endl;
        return;
    }
    else
    {
        // Initialisation
        size_t baseSize = src.getSize() * sizeof(unsigned char);
        int paddedWidth = src.getWidth() + 4 * radius;
        int paddedHeight = src.getHeight() + 4 * radius;
        size_t paddedSizeUC = paddedWidth * paddedHeight * src.getChannels() * sizeof(unsigned char);
        size_t paddedSizeF = paddedWidth * paddedHeight * src.getChannels() * sizeof(float);
        unsigned char* d_padded1, * d_padded2, * d_src, * d_src2, * d_dst;
        float* d_ak, * d_bk;
        clock_t timer = clock();

        // Memory allocation on device
        checkCudaErrors(cudaMalloc((void**)&d_src, baseSize));
        checkCudaErrors(cudaMalloc((void**)&d_src2, baseSize));
        checkCudaErrors(cudaMalloc((void**)&d_dst, baseSize));
        checkCudaErrors(cudaMalloc((void**)&d_padded1, paddedSizeUC));
        checkCudaErrors(cudaMalloc((void**)&d_padded2, paddedSizeUC));
        checkCudaErrors(cudaMalloc((void**)&d_ak, paddedSizeF));
        checkCudaErrors(cudaMalloc((void**)&d_bk, paddedSizeF));

        // Copy to device memory
        checkCudaErrors(cudaMemcpy(d_src, src.getData(), baseSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_src2, src.getData(), baseSize, cudaMemcpyHostToDevice));

        // Extand base array to have the correct result on borders
        replicate(d_padded1, d_src, src.getWidth(), src.getHeight(), 2 * radius, src.getChannels());

        // Run device function
        dim3 block_dim(32, 32);
        dim3 grid_dim((paddedWidth * src.getChannels() + block_dim.x - 1) / block_dim.x, (paddedHeight + block_dim.y - 1) / block_dim.y);
        guidedFirstKernel << <grid_dim, block_dim >> > (d_ak, d_bk, d_padded1, paddedWidth, paddedHeight, radius, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        guidedSecondKernel << <grid_dim, block_dim >> > (d_padded2, d_ak, d_bk, d_padded1, paddedWidth, paddedHeight, radius, src.getChannels());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Crop the result to get relevant data
        crop(d_src, d_padded2, 2 * radius, 2 * radius, src.getWidth(), src.getHeight(), paddedWidth, paddedHeight, src.getChannels());

        grid_dim.x = (src.getWidth() * src.getChannels() + block_dim.x - 1) / block_dim.x;
        grid_dim.y = (src.getHeight() + block_dim.y - 1) / block_dim.y;
        unsharpMaskingKernel << <grid_dim, block_dim >> > (d_dst, d_src2, d_src, dst.getWidth(), dst.getHeight(), value, dst.getChannels());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Retrieve result to host memory
        checkCudaErrors(cudaMemcpy(dst.getData(), d_dst, baseSize, cudaMemcpyDeviceToHost));
        std::cout << "Duration of guidedFilterEnhancement : " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds" << std::endl;

        // Free memory
        checkCudaErrors(cudaFree(d_src));
        checkCudaErrors(cudaFree(d_src2));
        checkCudaErrors(cudaFree(d_dst));
        checkCudaErrors(cudaFree(d_padded1));
        checkCudaErrors(cudaFree(d_padded2));
        checkCudaErrors(cudaFree(d_ak));
        checkCudaErrors(cudaFree(d_bk));
    }
}