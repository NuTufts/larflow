#ifndef __LARFLOW_RECO_NUSEL_VERTEX_VARS_H__
#define __LARFLOW_RECO_NUSEL_VERTEX_VARS_H__

#include "DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"


namespace larflow {
namespace reco {

  class NuSelVertexVars : public larcv::larcv_base {

  public:

    NuSelVertexVars()
      : larcv::larcv_base("NuSelVertexVars")
      {};
    virtual ~NuSelVertexVars() {};

    void analyze( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  larflow::reco::NuVertexCandidate& nuvtx,
                  larflow::reco::NuSelectionVariables& output );

    
  protected:
    

  };
  
}
}


#endif
