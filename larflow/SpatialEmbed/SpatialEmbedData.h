#ifndef __SPATIALEMBEDDATA_H__
#define __SPATIALEMBEDDATA_H__

#include <Python.h>
#include "bytesobject.h"

#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"
#include "larflow/SpatialEmbed/PrepMatchEmbed.h"

#include <map>
#include <vector>
#include <string>
#include <unordered_map>


namespace larcv {
  class Image2D;
  class IOManager;
}

namespace larlite {
  class event_mctrack;
  class event_mcshower;
  class event_mctruth;
  class storage_manager;
}

namespace ublarcvapp {
    namespace mctools {
        class MCPixelPGraph;
    }
}


namespace larflow {
  
namespace spatialembed {

class SpatialEmbedData{

public:

    SpatialEmbedData();
    ~SpatialEmbedData();

    struct CoordPix{
        int row;
        int col;
        int batch;
    };

    struct InstancePix{
        int row;
        int col;
    };

    void processImageData( larcv::EventImage2D* ev_adc, double threshold );

    void processLabelData(larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    void processLabelData( ublarcvapp::mctools::MCPixelPGraph* mcpg,
                           larflow::spatialembed::PrepMatchEmbed* prepembed );

    void processLabelDataWithShower( ublarcvapp::mctools::MCPixelPGraph* mcpg,
                            larflow::spatialembed::PrepMatchEmbed* prepembed,
                            larcv::EventImage2D* ev_adc );


    int num_instances_plane(int plane);


    std::vector<std::vector<int>> getTypes();
    std::vector<std::vector<CoordPix>> getCoord_t(); 
    std::vector<std::vector<double>> getFeat_t(); 
    std::vector<std::vector<std::vector<InstancePix>>> getInstances_t(); 


    PyObject* coord_t_pyarray(int plane);
    PyObject* feat_t_pyarray(int plane);
    PyObject* instance(int plane, int instance);
    PyObject* instance_binary(int plane, int instance);
    PyObject* get_instance_binaries(int plane);
    PyObject* get_class_map(int plane, int type, int include_opp=0); // include negatives aka 11 == -11
    PyObject* type_indices(int plane, int type, int include_opp=0);

    int typeof_instance(int plane, int instance);

    void check_instance_parity();

private:

    std::vector<std::vector<CoordPix>> coord_t; // vector length 3: coord_plane0, coord_plane1, coord_plane2. each coord_plane contains list of pixels
    std::vector<std::vector<double>> feat_t;  // vector length 3: feat_plane0, feat_plane1, feat_plane2. each feat_plane contains list of pixel values
    
    
    std::vector<std::vector<int>> types_t; // vector length 3, one for each plane
                                           // list of instance types
                                    
    std::vector<std::vector<std::vector<InstancePix>>> instances_t; // vector length 3, one for each plane
                                                                    // For each plane, 1d is instances
                                                                    // 2nd dimension is pixels per instance

                  
    std::vector<std::vector<std::vector<int>>> instances_binary_t; // vector length 3, one for each plane
                                                                    // For each plane, list of instance binary maps
                                                                    // each binary map has an entry for each non-zero pixel in coord_t for that plane
                                                                    // 0 for not part of that instance, 1 for part of it

    // std::vector<std::vector<std::vector<int>>> types_binary_t; // vector length 3, one for each plane
                                                            //    for each plane, vector size #instance classes in plane
                                                            //    each vector for instance class is a 0/1 map on whether or not
                                                            //    pixel at that location is of that class
                                                            //    (aka logical OR's instances of the same class into one vector)

    bool _setup_numpy;

    
};

}
}

    

#endif
