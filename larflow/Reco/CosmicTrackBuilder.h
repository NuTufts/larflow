#ifndef __LARFLOW_RECO_COSMIC_TRACK_BUILDER_H__
#define __LARFLOW_RECO_COSMIC_TRACK_BUILDER_H__

#include "larcv/core/Base/larcv_base.h"
#include "TrackClusterBuilder.h"

namespace larflow {
namespace reco {

  class CosmicTrackBuilder : public TrackClusterBuilder {

  public:

    CosmicTrackBuilder()
      {};
    virtual ~CosmicTrackBuilder() {};

    // override the process command
    // we use cosmic keypoint seeds to build tracks
    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    
  };
  
}
}

#endif
