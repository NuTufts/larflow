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
    instances_binary_t.clear();

    _setup_numpy = false; // reset

    std::vector<ublarcvapp::mctools::MCPixelPGraph::Node_t*> tids_from_neutrino 
        = mcpg->getNeutrinoPrimaryParticles();

    std::unordered_map<std::string, int> binary_hash_map;

    int num_instances = tids_from_neutrino.size();
    for (int plane = 0; plane < 3; plane++){

        std::vector<int> types;
        std::vector<std::vector<larflow::spatialembed::SpatialEmbedData::InstancePix>> instance_pixels;  // for sparse array
        std::vector<std::vector<int>> instance_binaries; // for binary maps


        for (int node = 0; node < num_instances; node++){ // loop over instances

            try{ 
                std::vector<larflow::spatialembed::AncestorIDPix_t> pixlist 
                    = prepembed->get_instance_pixlist(plane, tids_from_neutrino[node]->tid);
                
                types.push_back(tids_from_neutrino[node]->pid);  

                // Process pixel coords
                std::vector<larflow::spatialembed::SpatialEmbedData::InstancePix> pixels;
                int num_pixels = pixlist.size();
                for (int pixel = 0; pixel < num_pixels; pixel++ ){
                    // Create pixel object and add it to list
                    larflow::spatialembed::SpatialEmbedData::InstancePix pix {pixlist[pixel].row, pixlist[pixel].col};
                    pixels.push_back(pix);

                    // Indicate that that pixel is in an instance
                    binary_hash_map[std::to_string(pixlist[pixel].row) + "," + std::to_string(pixlist[pixel].col)] = 1;
                }
                instance_pixels.push_back(pixels);


                int pixel_count = 0;
                // Process binary map
                std::vector<int> instance_binary;
                int image_size = coord_t[plane].size();
                for (int pixel = 0; pixel < image_size; pixel++ ){
                    std::string image_pix_key = std::to_string(coord_t[plane][pixel].row) + "," + std::to_string(coord_t[plane][pixel].col);
                    if (binary_hash_map.find(image_pix_key) == binary_hash_map.end()){
                        instance_binary.push_back(0);
                    }
                    else {
                        instance_binary.push_back(binary_hash_map[image_pix_key]);
                        pixel_count++;
                    }
                }

                instance_binaries.push_back(instance_binary);
                binary_hash_map.clear();

                // std::cout << "Plane: " << plane << ", Instance: " << node << ", Coordsize: " << pixels.size() << ", Pixcount: " << pixel_count << std::endl;
                if (pixels.size() != pixel_count){
                    std::cerr << "NUMBER OF INSTANCE PIXELS MISMATCH WITH 1'S IN BINARY MAP" << std::endl;
                    exit(1);
                }

            } catch (const std::runtime_error& e){
                std::cout << "ID " << tids_from_neutrino[node]->tid << 
                          " does not exist in plane " << plane << " ";
                std::cout << "(" << e.what() << ")" << std::endl;
            }
        }
        types_t.push_back(types);
        instances_t.push_back(instance_pixels);
        instances_binary_t.push_back(instance_binaries);


    } 
}

void SpatialEmbedData::check_instance_parity(){

    for (int plane=0; plane < 3; plane++){
        int num_instances = instances_binary_t[plane].size();

        std::cerr << "Coord #inst, Binary #inst: " << instances_t[plane].size() << ", " << instances_binary_t[plane].size() << std::endl;
        for (int instance=0; instance < num_instances; instance++){
            int real_inst_count = instances_t[plane][instance].size();

            int binary_inst_count = 0;

            int num_points = instances_binary_t[plane][instance].size();
            for (int i = 0; i < num_points; i++){
                if (instances_binary_t[plane][instance][i] == 1){
                    binary_inst_count++;
                }
            }
            std::cerr << "Plane: " << plane << ", Instance: " << instance << ", Coord #: " << real_inst_count << ", Binary #: " << binary_inst_count << std::endl;
        }
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

    PyArrayObject* array = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT);

    for (int idx=0; idx < dims[0]; idx++){
        *((int*)PyArray_GETPTR2(array, idx, 0)) = coord_t[plane][idx].row;
        *((int*)PyArray_GETPTR2(array, idx, 1)) = coord_t[plane][idx].col;
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
    dims[1] = 1;

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
    
    PyArrayObject* array = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT);

    for (int idx=0; idx < dims[0]; idx++){
        *((int*)PyArray_GETPTR2(array, idx, 0)) = instances_t[plane][instance][idx].row;
        *((int*)PyArray_GETPTR2(array, idx, 1)) = instances_t[plane][instance][idx].col;
    }

    return (PyObject*) array;
}

PyObject* SpatialEmbedData::instance_binary(int plane, int instance){
    if ( !_setup_numpy ){
        import_array1(0);
        _setup_numpy = true;
    }
    if (plane > 2){
        std::string error = "Plane must be between [0,2]. Input was: " + std::to_string(plane) + ".\n";
        throw std::runtime_error(error);
    }
    if ((instances_binary_t[plane].size() != 0) && instance+1 > instances_binary_t[plane].size()){ //only if num instances != 0, in which case return empty array
        std::string error = "Requested instance " + std::to_string(instance) + " is outside number of instances " + std::to_string(instances_binary_t[plane].size()) 
                            + " in plane " + std::to_string(plane) + ".\n";
        throw std::runtime_error(error);
    }

    npy_intp* dims = new npy_intp[2];
    if (instances_binary_t[plane].size() == 0){
        dims[0] = 0;
        dims[1] = 1;
    } 
    else{
        dims[0] = instances_binary_t[plane][instance].size();
        dims[1] = 1;
    }
    
    PyArrayObject* array = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_INT);

    for (int idx=0; idx < dims[0]; idx++){
        *((int*)PyArray_GETPTR2(array, idx, 0)) = instances_binary_t[plane][instance][idx];
    }
    return (PyObject*) array;
}


PyObject* SpatialEmbedData::get_instance_binaries(int plane){
    if ( !_setup_numpy ){
        import_array1(0);
        _setup_numpy = true;
    }
    if (plane > 2){
        std::string error = "Plane must be between [0,2]. Input was: " + std::to_string(plane) + ".\n";
        throw std::runtime_error(error);
    }

    npy_intp* dims = new npy_intp[2];
    if (instances_binary_t[plane].size() == 0){
        dims[0] = 0;
        dims[1] = 1;
    } 
    else{
        dims[0] = instances_binary_t[plane].size();
        dims[1] = instances_binary_t[plane][0].size();
    }
    
    PyArrayObject* array = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT);

    for (int i=0; i < dims[0]; i++){
        for (int j=0; j < dims[1]; j++)
            *((int*)PyArray_GETPTR2(array, i, j)) = instances_binary_t[plane][i][j];
    }
    return (PyObject*) array;
}

PyObject* SpatialEmbedData::get_class_map(int plane, int type, int include_opp ){
    if ( !_setup_numpy ){
        import_array1(0);
        _setup_numpy = true;
    }
    if (plane > 2){
        std::string error = "Plane must be between [0,2]. Input was: " + std::to_string(plane) + ".\n";
        throw std::runtime_error(error);
    }

    std::vector<int> indices;
    int num_instances = instances_t[plane].size();
    for (int inst_type=0; inst_type < num_instances; inst_type++){
        if ((types_t[plane][inst_type] == type) || (include_opp && types_t[plane][inst_type] == -1*type)){
            indices.push_back(inst_type);
        }
    }

    npy_intp* dims = new npy_intp[2];
    if ((instances_binary_t[plane].size() == 0) || (indices.size() == 0)){
        dims[0] = coord_t[plane].size();
        dims[1] = 1;
    } 
    else{
        dims[0] = instances_binary_t[plane][0].size();
        dims[1] = 1;
    }
    
    PyArrayObject* array = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_INT);

    int is_true = 0;
    // loop for number of pixels
    for (int idx=0; idx < dims[0]; idx++){
        is_true = 0;
        for (int type_idx=0; type_idx < indices.size(); type_idx++){ // check instances of that type
            if (instances_binary_t[plane][indices[type_idx]][idx] == 1){
                is_true = 1;
                break;
            }   
        }
         *((int*)PyArray_GETPTR2(array, idx, 0)) = is_true;
    }

    return (PyObject*) array;
}


PyObject* SpatialEmbedData::type_indices(int plane, int type, int include_opp ){
    if ( !_setup_numpy ){
        import_array1(0);
        _setup_numpy = true;
    }
    if (plane > 2){
        std::string error = "Plane must be between [0,2]. Input was: " + std::to_string(plane) + ".\n";
        throw std::runtime_error(error);
    }

    std::vector<int> indices;
    int num_instances = instances_t[plane].size();
    for (int inst_type=0; inst_type < num_instances; inst_type++){
        if ((types_t[plane][inst_type] == type) || (include_opp && types_t[plane][inst_type] == -1*type)){
            indices.push_back(inst_type);
        }
    }

    npy_intp* dims = new npy_intp[2];
    if ((instances_binary_t[plane].size() == 0) || (indices.size() == 0)){
        dims[0] = 0;
        dims[1] = 1;
    } 
    else{
        dims[0] = indices.size();
        dims[1] = 1;
    }
    
    PyArrayObject* array = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_INT);

    for (int idx = 0; idx < dims[0]; idx++){
        *((int*)PyArray_GETPTR2(array, idx, 0)) = indices[idx];
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
