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

SpatialEmbedData::SpatialEmbedData() { }

SpatialEmbedData::~SpatialEmbedData() { }

}
}
