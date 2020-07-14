#include "SpatialEmbedData.h"
#include <iostream>
#include <string> 

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "ublarcvapp/MCTools/crossingPointsAnaMethods.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"

namespace larflow {
namespace spatialembed {

SpatialEmbedData::SpatialEmbedData( ) { 
    _setup_numpy = false;
}

SpatialEmbedData::~SpatialEmbedData() { }


void SpatialEmbedData::processImageData( larcv::EventImage2D* ev_adc, double threshold)
{
    coord_t.clear();
    feat_t.clear();
    _setup_numpy = false; // reset

    std::cout <<  "number of images: " << ev_adc->Image2DArray().size() << std::endl;
    std::vector<larcv::Image2D> adc_v = ev_adc->Image2DArray();

    // for plane in image data
    for (int i = 0; i < adc_v.size(); i++){
        larcv::Image2D plane_img = adc_v[i];
        std::cout << "image[" << i << "] " << plane_img.meta().dump() << std::endl;
        
        std::vector<larflow::spatialembed::SpatialEmbedData::CoordPix> coord_plane;
        std::vector<double> feat_plane;
        for (int row = 0; row < (int)plane_img.meta().rows(); row++){
            for (int col = 0; col < (int)plane_img.meta().cols(); col++){
                if (plane_img.pixel(row, col) > threshold){
                    larflow::spatialembed::SpatialEmbedData::CoordPix pix {row, col, 0};
                    coord_plane.push_back(pix);
                    feat_plane.push_back(plane_img.pixel(row, col));
                }
            }
        }
        coord_t.push_back(coord_plane);
        feat_t.push_back(feat_plane);
    }

}

void SpatialEmbedData::processLabelData(larcv::IOManager& iolcv, larlite::storage_manager& ioll ){
    
    ublarcvapp::mctools::MCPixelPGraph mcpg = ublarcvapp::mctools::MCPixelPGraph();

    mcpg.set_adc_treename( "wiremc" );
    mcpg.buildgraph( iolcv, ioll );

    larflow::PrepMatchTriplets preptriplet = larflow::PrepMatchTriplets();
    larflow::spatialembed::PrepMatchEmbed prepembed = larflow::spatialembed::PrepMatchEmbed();
    prepembed.process(iolcv, ioll, preptriplet);

    processLabelData(&mcpg, &prepembed);
}

void SpatialEmbedData::processLabelData( ublarcvapp::mctools::MCPixelPGraph* mcpg,
                           larflow::spatialembed::PrepMatchEmbed* prepembed )
{
    types_t.clear();
    instances_t.clear();
    _setup_numpy = false; // reset


    std::vector<ublarcvapp::mctools::MCPixelPGraph::Node_t*> tids_from_neutrino 
        = mcpg->getNeutrinoPrimaryParticles();

    int num_instances = tids_from_neutrino.size();
    for (int plane = 0; plane < 3; plane++){

        std::vector<int> types;
        std::vector<std::vector<larflow::spatialembed::SpatialEmbedData::InstancePix>> instance_pixels;

        for (int node = 0; node < num_instances; node++){ // loop over instances

            try{ 
                std::vector<larflow::spatialembed::AncestorIDPix_t> pixlist 
                    = prepembed->get_instance_pixlist(plane, tids_from_neutrino[node]->tid);
                
                types.push_back(tids_from_neutrino[node]->pid);  
                
                std::vector<larflow::spatialembed::SpatialEmbedData::InstancePix> pixels;
                int num_pixels = pixlist.size();
                for (int pixel = 0; pixel < num_pixels; pixel++ ){
                    larflow::spatialembed::SpatialEmbedData::InstancePix pix {pixlist[pixel].row, pixlist[pixel].col};
                    pixels.push_back(pix);
                }

                instance_pixels.push_back(pixels);
            } catch (const std::runtime_error& e){
                std::cout << "ID " << tids_from_neutrino[node]->tid << 
                          " does not exist in plane " << plane << " ";
                std::cout << "(" << e.what() << ")" << std::endl;
            }
        }
        types_t.push_back(types);
        instances_t.push_back(instance_pixels);
    }
}

PyObject* SpatialEmbedData::coord_t_pyarray(int plane){
    if ( !_setup_numpy ){
        import_array1(0);
        _setup_numpy = true;
    }
    if (plane > 2){
        std::string error = "Plane must be between [0,2]. Input was: " + std::to_string(plane) + ".\n";
        throw std::runtime_error(error);
    }

    npy_intp* dims = new npy_intp[2];
    dims[0] = coord_t[plane].size();
    dims[1] = 3;

    PyArrayObject* array = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT);

    for (int idx=0; idx < dims[0]; idx++){
        *((float*)PyArray_GETPTR2(array, idx, 0)) = coord_t[plane][idx].row;
        *((float*)PyArray_GETPTR2(array, idx, 1)) = coord_t[plane][idx].col;
        *((int*)PyArray_GETPTR2(array, idx, 2)) = coord_t[plane][idx].batch;
    }

    return (PyObject*) array;
}

PyObject* SpatialEmbedData::feat_t_pyarray(int plane){
    if ( !_setup_numpy ){
        import_array1(0);
        _setup_numpy = true;
    }
    if (plane > 2){
        std::string error = "Plane must be between [0,2]. Input was: " + std::to_string(plane) + ".\n";
        throw std::runtime_error(error);
    }

    npy_intp* dims = new npy_intp[2];
    dims[0] = feat_t[plane].size();
    dims[1] = 2;

    PyArrayObject* array = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT);

    for (int idx=0; idx < dims[0]; idx++){
        *((float*)PyArray_GETPTR2(array, idx, 0)) = feat_t[plane][idx];
    }

    return (PyObject*) array;
}

int SpatialEmbedData::num_instances_plane(int plane){
    if (plane > 2){
        std::string error = "Plane must be between [0,2]. Input was: " + std::to_string(plane) + ".\n";
        throw std::runtime_error(error);
    }
    return instances_t[plane].size();
}

int SpatialEmbedData::typeof_instance(int plane, int instance){
    if ( !_setup_numpy ){
        import_array1(0);
        _setup_numpy = true;
    }
    if (plane > 2){
        std::string error = "Plane must be between [0,2]. Input was: " + std::to_string(plane) + ".\n";
        throw std::runtime_error(error);
    }
    if ((instances_t[plane].size() != 0) && instance+1 > instances_t[plane].size()){ //only if num instances != 0, in which case return empty array
        std::string error = "Requested type at instance " + std::to_string(instance) + " is outside number of instances " + std::to_string(instances_t[plane].size()) 
                            + " in plane " + std::to_string(plane) + ".\n";
        throw std::runtime_error(error);
    }

    if (types_t[plane].size() == 0){
        return -1;
    }
    else{
        return types_t[plane][instance];
    }
}

PyObject* SpatialEmbedData::instance(int plane, int instance){
    if ( !_setup_numpy ){
        import_array1(0);
        _setup_numpy = true;
    }
    if (plane > 2){
        std::string error = "Plane must be between [0,2]. Input was: " + std::to_string(plane) + ".\n";
        throw std::runtime_error(error);
    }
    if ((instances_t[plane].size() != 0) && instance+1 > instances_t[plane].size()){ //only if num instances != 0, in which case return empty array
        std::string error = "Requested instance " + std::to_string(instance) + " is outside number of instances " + std::to_string(instances_t[plane].size()) 
                            + " in plane " + std::to_string(plane) + ".\n";
        throw std::runtime_error(error);
    }

    npy_intp* dims = new npy_intp[2];
    if (instances_t[plane].size() == 0){
        dims[0] = 0;
        dims[1] = 2;
    } 
    else{
        dims[0] = instances_t[plane][instance].size();
        dims[1] = 2;
    }
    
    PyArrayObject* array = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT);

    for (int idx=0; idx < dims[0]; idx++){
        *((float*)PyArray_GETPTR2(array, idx, 0)) = instances_t[plane][instance][idx].row;
        *((float*)PyArray_GETPTR2(array, idx, 1)) = instances_t[plane][instance][idx].col;
    }

    return (PyObject*) array;
}


std::vector<std::vector<larflow::spatialembed::SpatialEmbedData::CoordPix>> SpatialEmbedData::getCoord_t(){
    return coord_t;
} 

std::vector<std::vector<double>> SpatialEmbedData::getFeat_t(){
    return feat_t;
} 

std::vector<std::vector<std::vector<larflow::spatialembed::SpatialEmbedData::InstancePix>>> SpatialEmbedData::getInstances_t(){
    return instances_t;
} 

std::vector<std::vector<int>> SpatialEmbedData::getTypes(){
    return types_t;
}


}
}
