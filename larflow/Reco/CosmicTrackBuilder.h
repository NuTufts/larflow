#ifndef __LARFLOW_RECO_COSMIC_TRACK_BUILDER_H__
#define __LARFLOW_RECO_COSMIC_TRACK_BUILDER_H__

#include "larcv/core/Base/larcv_base.h"
#include "TrackClusterBuilder.h"

namespace larflow {
namespace reco {

  class CosmicTrackBuilder : public TrackClusterBuilder {

  public:

    CosmicTrackBuilder()
      : _do_boundary_analysis(false)
      {};
    virtual ~CosmicTrackBuilder() {};

    // override the process command
    // we use cosmic keypoint seeds to build tracks
    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void do_boundary_analysis( bool doit ) { _do_boundary_analysis = doit; };
    
  protected:

    bool _do_boundary_analysis;
    void _boundary_analysis( larlite::storage_manager& ioll );
    
  };
  
}
}

#endif
