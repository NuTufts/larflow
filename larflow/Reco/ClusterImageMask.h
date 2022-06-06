#ifndef __LARFLOW_RECO_CLUSTER_IMAGE_MASK_H__
#define __LARFLOW_RECO_CLUSTER_IMAGE_MASK_H__

#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/track.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"

#include "NuVertexCandidate.h"

namespace larflow {
namespace reco {

  class ClusterImageMask : public larcv::larcv_base {

  public:

    ClusterImageMask()
      : _npix(0)
      {};
    virtual ~ClusterImageMask() {};


    std::vector< larcv::Image2D >
      makeChargeMask( NuVertexCandidate& nuvtx,
                      const std::vector<larcv::Image2D>& adc_v );

    
    void maskCluster( const larlite::larflowcluster& cluster,
                      const std::vector<const larcv::Image2D*>& padc_v,
                      std::vector<larcv::Image2D>& mask_v,
                      const float thresh,
                      const int dpix=2 );

    void maskTrack( const larlite::track& track,
                    const std::vector<const larcv::Image2D*>& adc_v,
                    std::vector<larcv::Image2D>& mask_v,
		    const int tpcid, const int cryoid,		    
                    const float thresh,
                    const int dcol=2,
                    const int drow=2,
                    const float minstepsize=0.1,
                    const float maxstepsize=1.0 );

    void maskTrack( const larlite::track& track,
                    const std::vector<larcv::Image2D>& adc_v,
                    std::vector<larcv::Image2D>& mask_v,
		    const int tpcid, const int cryoid,
                    const float thresh,
                    const int dcol=2,
                    const int drow=2,
                    const float minstepsize=0.1,
                    const float maxstepsize=1.0 );
    
    int _npix;
    
  };

}
}


#endif
