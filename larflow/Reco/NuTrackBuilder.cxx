#include "NuTrackBuilder.h"

namespace larflow {
namespace reco {

  /**
   * @brief Run track builder on neutrino candidate tracks
   *
   * Using NuVertexCandidate instances as a seed, 
   * build out tracks by connecting cluster ends.
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   * @param[in] nu_candidate_v Neutrino proto-vertices produced by NuVertexMaker.
   */
  void NuTrackBuilder::process( larcv::IOManager& iolcv,
                                larlite::storage_manager& ioll,
                                const std::vector<NuVertexCandidate>& nu_candidate_v )
  {

    // clear segments, connections, proposals
    clear();

    // get clusters, pca-axis
    std::vector< std::string > cluster_producers =
      { "trackprojsplit_full",
        "trackprojsplit_wcfilter" };
    
    for ( auto const& producer : cluster_producers ) {
    
      larlite::event_larflowcluster* ev_cluster
        = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, producer);
      larlite::event_pcaxis* ev_pcaxis
        = (larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis,producer);      
      loadClusterLibrary( *ev_cluster, *ev_pcaxis );
      
    }
    
    buildNodeConnections();

    
    set_output_one_track_per_startpoint( true );    

    for (auto const& nuvtx : nu_candidate_v ) {
      buildTracksFromPoint( nuvtx.pos );
    }

    larlite::event_track* evout_track
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "nutrack");

    fillLarliteTrackContainer( *evout_track );
    
  }
  
}
}
