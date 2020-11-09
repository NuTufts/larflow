#ifndef __LARFLOW_RECO_TRACKDQDX_H__
#define __LARFLOW_RECO_TRACKDQDX_H__

#include <vector>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/track.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class TrackdQdx
   * @brief Calculates the dqdx for a given track cluster
   *
   */  

  class TrackdQdx : public larcv::larcv_base {

  public:

    TrackdQdx() {};
    ~TrackdQdx() {};
    
  public:
    
    larlite::track calculatedQdx( const larlite::track& track,
                                  const larlite::larflowcluster& trackhits,
                                  const std::vector<larcv::Image2D>& adc_v ) const;
    

  };


}
}
    

#endif
