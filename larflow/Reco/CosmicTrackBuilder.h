#ifndef __LARFLOW_RECO_COSMIC_TRACK_BUILDER_H__
#define __LARFLOW_RECO_COSMIC_TRACK_BUILDER_H__

#include "larcv/core/Base/larcv_base.h"
#include "TrackClusterBuilder.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class CosmicTrackBuilder
   * @brief Build cosmic tracks by connecting track clusters end-to-end
   *
   * Built on the class TrackClusterBuider, which treats the connection of
   * track clusters in a graph-traversal manner.
   *
   * This is dedicated to reconstructing cosmic muon tracks.
   * It doesn't aim so much for accuracy, but by trying to look for segments
   * which are plausibly attached to the edge of the detector.
   * This allows it to be used to veto candidate vertices as likely cosmic.
   *
   */
  class CosmicTrackBuilder : public TrackClusterBuilder {

  public:

    CosmicTrackBuilder()
      : _do_boundary_analysis(true)
      {};
    virtual ~CosmicTrackBuilder() {};

    // override the process command
    // we use cosmic keypoint seeds to build tracks
    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    /** @brief set flag that if true, runs the boundary analysis */
    void do_boundary_analysis( bool doit ) { _do_boundary_analysis = doit; };
    
  protected:

    bool _do_boundary_analysis; ///< if true, split found tracks into boundary and contained tracks
    
    void _boundary_analysis_wflash( larlite::storage_manager& ioll );
    void _boundary_analysis_noflash( larlite::storage_manager& ioll );    
    
  };
  
}
}

#endif
