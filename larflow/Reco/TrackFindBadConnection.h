#ifndef __LARFLOW_RECO_FIND_BAD_CONNECTION_H__
#define __LARFLOW_RECO_FIND_BAD_CONNECTION_H__

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/DataFormat/track.h"

#include "NuVertexCandidate.h"

namespace larflow {
namespace reco {

  /**
   * @class TrackFindBadConnection
   * @ingroup Reco
   * @brief Look for segments that should not have been merged. Break if bad connection found.
   *
   */
  
  class TrackFindBadConnection : public larcv::larcv_base {
  public:

    TrackFindBadConnection()
      : larcv::larcv_base("TrackFindBadConnection")
      {};
    virtual ~TrackFindBadConnection() {};
    
    std::vector< larlite::track >
      splitBadTrack( const larlite::track& track,
                     const std::vector<larcv::Image2D>& adc_v,
                     float  mingap );
    
    int processNuVertexTracks( larflow::reco::NuVertexCandidate& nuvtx,
                               larcv::IOManager& iolcv );
    
    
  };

}
}


#endif
