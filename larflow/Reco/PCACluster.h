#ifndef __LARFLOW_PCA_CLUSTER_H__
#define __LARFLOW_PCA_CLUSTER_H__

#include <vector>

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflowcluster.h"

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
    
    int merge_clusters( std::vector<cluster_t>& cluster_v,
                        const std::vector<larcv::Image2D>& adc_v,
                        float max_dist_cm, float min_angle_deg, float max_pca2,
                        bool print_tests=false );

    void defragment_clusters( std::vector<cluster_t>& cluster_v,
                              const float max_2nd_pca_eigenvalue );

    larlite::larflowcluster makeLArFlowCluster( cluster_t& cluster,
                                                const std::vector<larcv::Image2D>& ssnet_showerimg_v,
                                                const std::vector<larcv::Image2D>& ssnet_trackimg_v );
    
    
  };

}
}

#endif
