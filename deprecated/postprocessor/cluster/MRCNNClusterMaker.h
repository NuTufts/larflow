#ifndef __MRCNN_CLUSTER_MAKER__
#define __MRCNN_CLUSTER_MAKER__

#include <vector>
#include "DataFormat/larflowcluster.h"
#include "DataFormat/larflow3dhit.h"

#include "larcv/core/DataFormat/ClusterMask.h"
#include "larcv/core/DataFormat/Image2D.h"

namespace larflow {
  
  class MRCNNClusterMaker  {
    
  public:
    
    MRCNNClusterMaker() {};
    virtual ~MRCNNClusterMaker() {};
    
    std::vector< larlite::larflowcluster > makeSimpleClusters( const std::vector<larcv::ClusterMask>&  masks_v,
                                                               const std::vector<larlite::larflow3dhit>& hits_v,
                                                               const std::vector<larcv::Image2D>& wholeview_adc_v );
    
    
  };
  
}

#endif
