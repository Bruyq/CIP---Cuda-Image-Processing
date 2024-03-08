#include "image.h"

/* Image object constructor
* Loads an image from file and create an Image object
* Parameter : filename (char_ptr)
*/
Image::Image(char* filename)
{
    m_data = stbi_load(filename, &m_width, &m_height, &m_channels, 3);
    if (m_data == NULL)
    {
        printf("Error in loading the image\n");
        exit(1);
    }
}

/* Image object constructor
* Creates an Image object
* Parameter : width (int), height (int), number of channels (int)
*/
Image::Image(int width, int height, int channels)
{
    m_width = width;
    m_height = height;
    m_channels = channels;
    m_data = new unsigned char[m_width * m_height * m_channels];
}

/* Image object destructor
*/
Image::~Image()
{
    delete[] m_data;
}

/* Image getWidth method
* Gives image width member (int)
*/
int Image::getWidth()
{
    return m_width;
}

/* Image getHeight method
* Gives image height member (int)
*/
int Image::getHeight()
{
    return m_height;
}

/* Image getChannels method
* Gives image number of channels member (int)
*/
int Image::getChannels()
{
    return m_channels;
}

/* Image getSize method
* Gives image size (int) where size is width x height x number of channels
*/
int Image::getSize()
{
    return m_height * m_width * m_channels;
}

/* Image printData method
* Write the data to a std::cout output
*/
void Image::printData()
{
    for (int j = 0; j < m_height; j++)
    {
        for (int i = 0; i < m_width; i++)
        {
            std::cout << static_cast<int>(m_data[(j * m_width + i) * m_channels]) << " "
                << static_cast<int>(m_data[(j * m_width + i) * m_channels + 1]) << " "
                    << static_cast<int>(m_data[(j * m_width + i) * m_channels + 2]) << "\n";
        }
    }
    return;
}

/* Image saveDataInPPM method
* Write the data to a ppm file
* Parameter : filename (const char*)
*/
void Image::saveDataInPPM(const char* filename)
{
    std::ofstream ofs(filename);
    ofs << "P3\n" << m_width << " " << m_height << "\n255\n";
    for (int j = 0; j < m_height; j++)
    {
        for (int i = 0; i < m_width; i++)
        {
            ofs << static_cast<int>(m_data[(j * m_width + i) * m_channels]) << " "
                << static_cast<int>(m_data[(j * m_width + i) * m_channels + 1]) << " "
                << static_cast<int>(m_data[(j * m_width + i) * m_channels + 2]) << "\n";
        }
    }
    ofs.close();
    return;
}

/* Image getData method
* Gives image pointer to the data (unsigned char)
*/
unsigned char* Image::getData()
{
    return m_data;
}

/* Image save method
* Saves the image under the filename specified (png format)
* Parameter : filename (char_ptr)
*/
void Image::save(const char* filename)
{
    stbi_write_png(filename, m_width, m_height, m_channels, m_data, m_width * m_channels); 
    // Sometimes this function isn't working properly, 1 channel image or image with low dimensions.
    // Why ? Tested without any kernel applied, so the problem comes from this routine
}