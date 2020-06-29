#ifndef __PrepSpatialEmbed__
#define __PrepSpatialEmbed__

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

class PrepSpatialEmbed{
private:
    TTree* img_tree;

public:
    PrepSpatialEmbed();
    ~PrepSpatialEmbed();
    
    void processTrainSet( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    void writeTrainSet();
    
    TTree* getTTree();

    void insertBranch(); 

};

}
}

    

#endif
