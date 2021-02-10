#ifndef __LARFLOW_RECO_NUSEL_PRONG_VARS_H__
#define __LARFLOW_RECO_NUSEL_PRONG_VARS_H__

#include "larcv/core/Base/larcv_base.h"
#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"

namespace larflow {
namespace reco {

  class NuSelProngVars : public larcv::larcv_base {

  public:

    NuSelProngVars()
      : larcv::larcv_base("NuSelProngVars")
      {};
    virtual ~NuSelProngVars() {};

    void analyze( larflow::reco::NuVertexCandidate& nuvtx,
                  larflow::reco::NuSelectionVariables& output );

  protected:

    int   _min_shower_nhits;
    float _min_shower_length;
    int   _min_track_nhits;
    float _min_track_length;
    
  };
  
}
}
    

#endif
