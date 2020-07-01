#include "SpatialEmbedData.h"
#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "ublarcvapp/MCTools/crossingPointsAnaMethods.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"


namespace larflow {
namespace spatialembed {

SpatialEmbedData::SpatialEmbedData( larcv::IOManager& iolcv_in, larlite::storage_manager& ioll_in ) 
{
    // iolcv = iolcv_in;
    // ioll  = ioll_in;
}

SpatialEmbedData::~SpatialEmbedData() { }

void SpatialEmbedData::processImageData()
{
    std::cout << "processImageData" << endl;
}

void SpatialEmbedData::processLabelData()
{
    std::count << "processLabelData()" << endl;
}
}
}
