#ifndef __LARFLOW_RECO_NUTRACK_BUILDER_H__
#define __LARFLOW_RECO_NUTRACK_BUILDER_H__

#include "TrackClusterBuilder.h"
#include "NuVertexMaker.h"

namespace larflow {
namespace reco {

  /** 
   * @ingroup NuTrackBuilder
   * @class NuTrackBuilder
   * @brief Build tracks by assembling clusters, starting from neutrino vertices
   *
   * Inherits from TrackClusterBuilder. The base class provides the track buiding algorithms.
   * This class provides interface to the NuVertexCandidate inputs.
   *
   */
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
