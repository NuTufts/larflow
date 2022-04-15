#ifndef __LARFLOW_RECO_NUTRACK_DQDX_H__
#define __LARFLOW_RECO_NUTRACK_DQDX_H__

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "NuVertexCandidate.h"


namespace larflow {
namespace reco {

  class NuTrackdQdx : public larcv::larcv_base {

  public:
    NuTrackdQdx()
      : larcv::larcv_base("NuTrackdQdx")
      {};
    virtual ~NuTrackdQdx() {};
    
    int process_nuvertex_tracks( larcv::IOManager& iolcv,
				 larflow::reco::NuVertexCandidate& nuvtx );
    
  };
}
}

#endif
