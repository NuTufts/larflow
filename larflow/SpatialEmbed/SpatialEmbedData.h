#ifndef __SPATIALEMBEDDATA_H__
#define __SPATIALEMBEDDATA_H__

#include <Python.h>
#include "bytesobject.h"

#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"
#include "larflow/SpatialEmbed/PrepMatchEmbed.h"

#include <map>
#include <vector>
#include <string>


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

    int num_instances_0;
    int num_instances_1;
    int num_instances_2;  

    void num_instances_change(int num);

    void processImageData( larcv::EventImage2D* ev_adc, double threshold );


    void processLabelData(larcv::IOManager& iolcv, larlite::storage_manager& ioll );


    void processLabelData( ublarcvapp::mctools::MCPixelPGraph* mcpg,
                           larflow::spatialembed::PrepMatchEmbed* prepembed );

    int num_instances_plane(int plane);


    std::vector<std::vector<int>> getTypes();
    std::vector<std::vector<CoordPix>> getCoord_t(); 
    std::vector<std::vector<double>> getFeat_t(); 
    std::vector<std::vector<std::vector<InstancePix>>> getInstances_t(); 

    PyObject* coord_t_pyarray(int plane);
    PyObject* feat_t_pyarray(int plane);
    PyObject* instance(int plane, int instance);
    int type(int plane, int instance);


private:

    std::vector<std::vector<CoordPix>> coord_t; // vector length 3: coord_plane0, coord_plane1, coord_plane2. each coord_plane contains list of pixels
    std::vector<std::vector<double>> feat_t;  // vector length 3: feat_plane0, feat_plane1, feat_plane2. each feat_plane contains list of pixel values
    
    std::vector<std::vector<int>> types_t; // vector length 3, one for each plane
                                           // list of instance types
                                    
    std::vector<std::vector<std::vector<InstancePix>>> instances_t; // vector length 3, one for each plane
                                                                    // For each plane, 1d is instances
                                                                    // 2nd dimension is pixels per instance

    bool _setup_numpy;

    
};

}
}

    

#endif
