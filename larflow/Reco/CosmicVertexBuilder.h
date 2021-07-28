#ifndef __LARFLOW_RECO_COSMIC_VERTEX_BUILDER_H__
#define __LARFLOW_RECO_COSMIC_VERTEX_BUILDER_H__

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "NuVertexCandidate.h"

namespace larflow {
namespace reco {

  class CosmicVertexBuilder : public larcv::larcv_base {
  public:

    CosmicVertexBuilder()
      : larcv::larcv_base("CosmicVertexBuilder") {};
    virtual ~CosmicVertexBuilder() {};
    
    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  std::vector<NuVertexCandidate>& nu_candidate_v );    

  };
  
}
}

#endif
