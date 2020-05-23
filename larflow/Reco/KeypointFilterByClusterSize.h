#ifndef __Keypoint_Filter_By_Cluster_Size_h__
#define __Keypoint_Filter_By_Cluster_Size_h__

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "DataFormat/storage_manager.h"

namespace larflow {
namespace reco {

  class KeypointFilterByClusterSize : public larcv::larcv_base {

  public:

    KeypointFilterByClusterSize()
      : larcv::larcv_base("KeypointFilterByClusterSize")
      {};
    virtual ~KeypointFilterByClusterSize() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    
    
  };

}
}


#endif
