#ifndef __LARFLOW_RECO_SHOWERGAP_ANA_2D_H__
#define __LARFLOW_RECO_SHOWERGAP_ANA_2D_H__

#include "DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"

namespace larflow {
namespace reco {

  

  class NuSelShowerGapAna2D : public larcv::larcv_base {

  public:

  NuSelShowerGapAna2D()
    : larcv::larcv_base("NuSelShowerGapAna2D")
      {};
    
    virtual ~NuSelShowerGapAna2D() {};

    void analyze( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  larflow::reco::NuVertexCandidate& nuvtx,
                  larflow::reco::NuSelectionVariables& output );
    

  };
  
}
}

#endif
