#include "histogram.h"

Histogram::Histogram(unsigned int nb_bins, unsigned int nb_channels) : m_bins(nb_bins), m_channels(nb_channels)
{
    m_hist = new unsigned int[m_bins * m_channels];
}

Histogram::~Histogram()
{
    delete[] m_hist;
}

unsigned int* Histogram::getHist()
{
    return m_hist;
}

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