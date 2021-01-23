#ifndef __LARFLOW_MASK_RCNN_RECO_H__
#define __LARFLOW_MASK_RCNN_RECO_H__

#include <vector>
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"

namespace larflow {
namespace mrcnnreco {

  /**
   * @class MaskRCNNreco
   * @brief Cluster larmatch points using MaskRCNN
   * 
   * Use Mask-RCNN to help cluster larmatch points into cosmic (and neutrino cluster).
   * Integrate keypoints and ssnet into points.
   * Attempt track reco as well.
   */
  class MaskRCNNreco : public larcv::larcv_base {
  public:
    MaskRCNNreco()
      : larcv::larcv_base("MaskRCNNreco")
      {
      };
    virtual ~MaskRCNNreco() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    std::vector<larcv::ClusterMask> merge_proposals( const std::vector<larcv::ClusterMask>& mask_v );
    
    std::vector<larlite::larflowcluster> clusterbyproposals( const larlite::event_larflow3dhit& ev_larmatch,
                                                             const std::vector<larcv::ClusterMask>& mask_v,
                                                             const std::vector<larcv::Image2D>& adc_v,                                                             
                                                             const float hit_threshold );
    

  protected:
    
  };
  
}
}

#endif
