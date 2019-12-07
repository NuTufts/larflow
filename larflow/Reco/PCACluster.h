#ifndef __LARFLOW_PCA_CLUSTER_H__
#define __LARFLOW_PCA_CLUSTER_H__

#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {
  
  class PCACluster {
  public:

    PCACluster() {};
    virtual ~PCACluster() {};

    void process( larcv::IOManager& iolc, larlite::storage_manager& ioll );


  protected:
    
    int split_clusters( std::vector<cluster_t>& cluster_v,
                        const std::vector<larcv::Image2D>& adc_v );

    
  };

}
}

#endif
