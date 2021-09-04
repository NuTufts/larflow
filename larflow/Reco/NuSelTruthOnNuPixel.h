#ifndef __LARFLOW_RECO_NUSEL_TRUTH_ON_NUPIXEL_H__
#define __LARFLOW_RECO_NUSEL_TRUTH_ON_NUPIXEL_H__

#include "larlite/DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"

namespace larflow {
namespace reco {

  class NuSelTruthOnNuPixel : public larcv::larcv_base {
  public:

    NuSelTruthOnNuPixel()
      : larcv::larcv_base("NuSelTruthOnPixel")
      {};
    virtual ~NuSelTruthOnNuPixel() {};

    void analyze( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  larflow::reco::NuVertexCandidate& nuvtx,
                  larflow::reco::NuSelectionVariables& output );
      

  };
  
}
}

#endif
