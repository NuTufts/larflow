#ifndef __LARFLOW_RECO_TRUTH_NU_VERTEX_MAKER_H__
#define __LARFLOW_RECO_TRUTH_NU_VERTEX_MAKER_H__

#include "larcv/core/DataFormat/larcv_base.h"

namespace larflow {
namespace reco {

  /**
   * 
   * @brief Produce a NuVertexCandidate from truth information and larmatch points
   *
   */
  class TruthNuVertexMaker : public larcv::larcv_base {

  public:

    TruthNuVertexMaker() {};
    virtual ~TruthNuVertexMaker() {};
    
  };
  
}
}

#endif
