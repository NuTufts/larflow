#include "CosmicTrackBuilder.h"

#include "DataFormat/larflow3dhit.h"
#include "DataFormat/track.h"

namespace larflow {
namespace reco {

  void CosmicTrackBuilder::process( larcv::IOManager& iolcv,
                                    larlite::storage_manager& ioll )
  {

    // get clusters, pca-axis
    std::string producer = "trackprojsplit_full";
    
    larlite::event_larflowcluster* ev_cluster
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, producer);
    larlite::event_pcaxis* ev_pcaxis
      = (larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis,producer);

    loadClusterLibrary( *ev_cluster, *ev_pcaxis );
    buildConnections();

    // get keypoints
    std::string producer_keypoint = "keypointcosmic";
    larlite::event_larflow3dhit* ev_keypoint
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, producer_keypoint );


    for (size_t ikp=0; ikp<ev_keypoint->size(); ikp++) {
      auto const& kp = ev_keypoint->at(ikp);
      std::vector<float> startpt = { kp[0], kp[1], kp[2] };
      buildTracksFromPoint( startpt );
    }

    larlite::event_track* evout_track
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "cosmictrack");

    fillLarliteTrackContainer( *evout_track );
    
  }
  
}
}
