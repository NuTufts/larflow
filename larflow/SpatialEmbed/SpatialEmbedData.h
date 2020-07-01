#ifndef __SpatialEmbedData__
#define __SpatialEmbedData__

#include <Python.h>
#include "bytesobject.h"

#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"

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

struct InstancePix{
    int row;
    int col;
};

class SpatialEmbedData{
private:
    std::vector<int> coord_plane0_t;
    std::vector<int> feat_plane0_t;
    std::vector<int> coord_plane1_t;
    std::vector<int> feat_plane1_t;
    std::vector<int> coord_plane2_t;
    std::vector<int> feat_plane3_t;
    
    std::vector<int> types_plane0;
    std::vector<std::vector<InstancePix>> instances_plane0;

    std::vector<int> types_plane1;
    std::vector<std::vector<InstancePix>> instances_plane1;

    std::vector<int> types_plane2;
    std::vector<std::vector<InstancePix>> instances_plane2;

    larcv::IOManager& iolcv;
    larlite::storage_manager& ioll;

public:
    SpatialEmbedData( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    ~SpatialEmbedData();

    void processImageData();
    void processLabelData();
    
};

}
}

    

#endif
