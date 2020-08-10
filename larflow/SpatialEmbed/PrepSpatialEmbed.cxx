#include "PrepSpatialEmbed.h"
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

PrepSpatialEmbed::PrepSpatialEmbed() 
{
    img_tree = new TTree("trainingdata", "Spatial Embed Centroid Training Data");
}

PrepSpatialEmbed::~PrepSpatialEmbed() 
{
    // if (img_tree) delete img_tree;
}


void PrepSpatialEmbed::processTrainSet( larcv::IOManager& iolcv, larlite::storage_manager& ioll ) {}
void PrepSpatialEmbed::writeTrainSet() {}

void PrepSpatialEmbed::insertBranch() 
{
    std::vector<int> temp_vect;
    temp_vect.push_back(1);
    temp_vect.push_back(2);
    temp_vect.push_back(3);

    for (int x : temp_vect){
        std::cout << x << " ";
    }

    img_tree->Branch("test", &temp_vect, "test/I");

    if (img_tree)
        img_tree->Write();

}

TTree* PrepSpatialEmbed::getTTree()
{
    return img_tree;
}

}
}