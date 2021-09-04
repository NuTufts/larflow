#ifndef __LARFLOW_RECO_NUSEL_WCTAGGER_OVERLAP_H__
#define __LARFLOW_RECO_NUSEL_WCTAGGER_OVERLAP_H__

#include <vector>

#include "larlite/DataFormat/track.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"

namespace larflow {
namespace reco {

  class NuSelWCTaggerOverlap : public larcv::larcv_base {

  public:

    NuSelWCTaggerOverlap()
      : larcv::larcv_base("NuSelWCTaggerOverlap")
      {};

    virtual ~NuSelWCTaggerOverlap() {};

    void analyze( larflow::reco::NuVertexCandidate& nuvtx,
                  larflow::reco::NuSelectionVariables& output,
                  larcv::IOManager& iolcv );
    

  };

}
}

#endif
