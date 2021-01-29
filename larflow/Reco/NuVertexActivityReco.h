#ifndef __LARFLOW_RECO_NUVERTEX_ACTIVITYRECO_H__
#define __LARFLOW_RECO_NUVERTEX_ACTIVITYRECO_H__

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class NuVertexActivityReco
   *
   * Uses larmatch points to find 3D-consistent point-like deposition.
   * Filter out background and real candiates using larmatch keypoint candidates.
   * Emphasis is end of showers.
   *
   */
  class NuVertexActivityReco : public larcv::larcv_base {
    
  public:
    NuVertexActivityReco()
      : larcv::larcv_base("NuVertexActivityReco") {};
    virtual ~NuVertexActivityReco() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

  protected:
    
    void makeClusters( larlite::storage_manager& ioll,
                       std::vector<larflow::reco::cluster_t>& cluster_v,
                       const float larmatch_threshold );

    std::vector<larlite::larflow3dhit>
      findVertexActivityCandidates( larlite::storage_manager& ioll,
                                    larcv::IOManager& iolcv,
                                    std::vector<larflow::reco::cluster_t>& cluster_v,
                                    const float va_threshold );
      
    
    std::vector<float> calcPlanePixSum( const larlite::larflow3dhit& hit,
                                        const std::vector<larcv::Image2D>& adc_v );
    
  };
}
}

#endif
