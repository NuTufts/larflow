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
   * @brief internal class to represent masks
   *
   */
  class Mask_t {
  public:
    Mask_t() {};
    Mask_t( const larcv::ClusterMask& mask, const larcv::ImageMeta& embedding_image );
    virtual ~Mask_t() {};
    larcv::Image2D img;      ///< global mask
    std::vector<int> pixcol; ///< col coordinate w/r/t embedding img
    std::vector<int> pixrow; ///< row coordinate w/r/t embedding img
    int col_range[2]; ///< col bounds
    int row_range[2]; ///< row bounds
  };

  
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

    std::vector<larflow::mrcnnreco::Mask_t>
      merge_proposals( std::vector<larflow::mrcnnreco::Mask_t>& mask_v );
    
    std::vector<larlite::larflowcluster> clusterbyproposals( const larlite::event_larflow3dhit& ev_larmatch,
                                                             const std::vector<larflow::mrcnnreco::Mask_t>& mask_v,
                                                             const float hit_threshold );
    

  protected:

    std::vector< larflow::mrcnnreco::Mask_t > _mask_v;
    
  };
  
}
}

#endif
