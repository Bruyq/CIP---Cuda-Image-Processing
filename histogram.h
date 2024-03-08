#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <iostream>
#include <fstream>


class Histogram
{
private:
	unsigned int* m_hist;
	unsigned int m_channels;
	unsigned int m_bins;

public:
	Histogram(unsigned int nb_bins, unsigned int nb_channels);
	~Histogram();
	unsigned int* getHist();
	void saveHistogram(const char* filename);
	void printHistogram();
};

#endif // HISTOGRAM_H