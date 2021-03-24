#ifndef __NUSEL_UNRECO_CHARGE_H__
#define __NUSEL_UNRECO_CHARGE_H__

#include "DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class NuSelUnrecoCharge
   * @brief Provides measure for the amount of unreconstructed charge around the
   *        neutrino candidate. Helps remove higher energy BNB-nu events 
   *        for Gen-2 nu-e selection.
   */
  
  class NuSelUnrecoCharge : public larcv::larcv_base {

  public:

    NuSelUnrecoCharge()
      : larcv::larcv_base("NuSelUnrecoCharge")
      {};
    virtual ~NuSelUnrecoCharge() {};

    void analyze( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  larflow::reco::NuVertexCandidate& nuvtx,
                  larflow::reco::NuSelectionVariables& output );    
    

  };
  
}
}

#endif
