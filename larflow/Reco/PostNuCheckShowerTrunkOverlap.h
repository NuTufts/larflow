#ifndef __LARFLOW_RECO_POSTNUCHECK_SHOWERTRUNK_OVERLAP_H__
#define __LARFLOW_RECO_POSTNUCHECK_SHOWERTRUNK_OVERLAP_H__

#include <vector>
#include "larcv/core/Base/larcv_base.h"
#include "NuVertexCandidate.h"

namespace larflow {
namespace reco {

  class PostNuCheckShowerTrunkOverlap : public larcv::larcv_base {
  public:
    
    PostNuCheckShowerTrunkOverlap()
      : larcv::larcv_base("PostNuCheckShowerTrunkOverlap")
    {};
    virtual ~PostNuCheckShowerTrunkOverlap() {};
    
    void process( std::vector<larflow::reco::NuVertexCandidate>& nu_v );
    
      
  };

}
}

#endif
