#include "histogram.h"

// Histogram object constructor
// Creates an histogram
// Parameter : 
// "nb_bins" for the number of histogram bins
// "nb_channels" for the number of channels of the corresponding data
Histogram::Histogram(unsigned int nb_bins, unsigned int nb_channels) : m_bins(nb_bins), m_channels(nb_channels)
{
    m_hist = new unsigned int[m_bins * m_channels];
}


// Histogram object destructor
// Deletes an histogram
Histogram::~Histogram()
{
    delete[] m_hist;
}


// Returns histogram data
unsigned int* Histogram::getHist()
{
    return m_hist;
}


// Saves an histogram in a text file
// Parameter : 
// "filename" for the corresponding text file filename (overrides if already exists)
void Histogram::saveHistogram(const char* filename)
{
    std::ofstream ofs(filename);
    ofs << "Histogramme de l'image " << filename << "\n";
    for (int c = 0; c < m_channels; c++)
    {
        for (int j = 0; j < m_bins; j++)
        {
            ofs << "Canal_" << c + 1 << "[" << j << "] = " << m_hist[j + m_bins * c] << "\n";
        }
    }
    ofs.close();
    return;
}


// Prints histogram data in the standard output
void Histogram::printHistogram()
{
    std::cout << "Histogramme de l'image selectionnee" << "\n";
    for (int c = 0; c < m_channels; c++)
    {
        for (int j = 0; j < m_bins; j++)
        {
            std::cout << "Canal_" << c + 1 << "[" << j << "] = " << m_hist[j + m_bins * c] << "\n";
        }
    }
}