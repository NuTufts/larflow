#ifndef __LARFLOW_RECO_TRACK_FORWARD_BACKWARD_LL_H__
#define __LARFLOW_RECO_TRACK_FORWARD_BACKWARD_LL_H__

#include "TGraphErrors.h"
#include "larcv/core/Base/larcv_base.h"

#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"

namespace larflow {
namespace reco {

  class TrackForwardBackwardLL : public larcv::larcv_base
  {
    
  public:

  TrackForwardBackwardLL()
    : larcv::larcv_base("TrackForwardBackwardLL")
      {};
    virtual ~TrackForwardBackwardLL() {};

    struct Pt_t {
      int ipt;      ///< index of point in track
      float x;      ///< position along track
      float dqdx;   ///< dq/dx value
      float ddqdx2; ///< derivative
      float var;    ///< variance of smoothed value
    };
    typedef std::vector<Pt_t> Track_t;
    
    void analyze( larflow::reco::NuVertexCandidate& nuvtx,
                  larflow::reco::NuSelectionVariables& nusel );

    std::vector< std::vector<TGraphErrors> > graph_vv; ///< for debugging

    Track_t _smooth( Track_t& track, const float maxdist, const int nn );    
    
  };
  
}
}


#endif
