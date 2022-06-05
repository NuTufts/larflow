#ifndef __LARFLOW_RECO_NUTRACK_BUILDER_H__
#define __LARFLOW_RECO_NUTRACK_BUILDER_H__

#include "TrackClusterBuilder.h"
#include "NuVertexMaker.h"
#include "ClusterBookKeeper.h"

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
                  std::vector<NuVertexCandidate>& nu_candidate_v,
		  std::vector<ClusterBookKeeper>& nu_cluster_book_v,
		  bool load_clusters=true );

    void loadClustersAndConnections( larcv::IOManager& iolcv,
				     larlite::storage_manager& ioll,
				     const int tpc,
				     const int cryoid );
    
    void set_verbosity( larcv::msg::Level_t v ) { TrackClusterBuilder::set_verbosity(v); };


    void _veto_assigned_clusters( ClusterBookKeeper& nuvtx_cluster_book );
    void _book_used_clusters( ClusterBookKeeper& nuvtx_cluster_book );
    
  };

}
}


#endif
