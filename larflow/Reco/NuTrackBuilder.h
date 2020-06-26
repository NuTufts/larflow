#ifndef __LARFLOW_RECO_NUTRACK_BUILDER_H__
#define __LARFLOW_RECO_NUTRACK_BUILDER_H__

#include "TrackClusterBuilder.h"
#include "NuVertexMaker.h"

namespace larflow {
namespace reco {

  class NuTrackBuilder : public TrackClusterBuilder {

  public:

    NuTrackBuilder() {};
    virtual ~NuTrackBuilder() {};


    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  const std::vector<NuVertexCandidate>& nu_candidate_v );

  };

}
}


#endif
