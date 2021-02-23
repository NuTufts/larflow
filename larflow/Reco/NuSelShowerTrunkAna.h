#ifndef __LARFLOW_RECO_NUSEL_SHOWERTRUNK_ANA_H__
#define __LARFLOW_RECO_NUSEL_SHOWERTRUNK_ANA_H__

#include <vector>

#include "DataFormat/track.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"

namespace larflow {
namespace reco {

  class NuSelShowerTrunkAna : public larcv::larcv_base {

  public:

    NuSelShowerTrunkAna()
      : larcv::larcv_base("NuSelShowerTrunkAna")
      {};

    virtual ~NuSelShowerTrunkAna() {};

    void analyze( larflow::reco::NuVertexCandidate& nuvtx,
                  larflow::reco::NuSelectionVariables& output,
                  larcv::IOManager& iolcv );
    
    std::vector<larlite::track>       _shower_dqdx_v;
    std::vector< std::vector<float> > _shower_avedqdx_v;
    std::vector< std::vector<float> > _shower_ll_v;
    std::vector<float>                _shower_gapdist_v;    
    
  };
  
}
}


#endif
