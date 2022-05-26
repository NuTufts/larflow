#include "EventKeypointReco.h"
#include "larflow/LArFlowConstants/LArFlowConstants.h"

namespace larflow {
namespace reco {

  EventKeypointReco::EventKeypointReco()
    : larcv::larcv_base("EventKeypointReco")
  {
    // set defaults
    kpalgo.set_sigma( 10.0 );
    kpalgo.set_min_cluster_size(   20.0, 0 );
    kpalgo.set_keypoint_threshold( 0.25, 0 );
    kpalgo.set_min_cluster_size(   20.0, 1 );    
    kpalgo.set_keypoint_threshold( 0.5, 1 );    
    kpalgo.set_larmatch_threshold( 0.3 );
    kpalgo.set_output_tree_name( "keypoint" );    
  }
  
  void EventKeypointReco::process_larmatch_v2( larlite::storage_manager& ioll,
					       std::string hittree )
  {

    // do what is done in KPS reco manager::recokeyoints
    // in KPS reco, we split cosmic and neutrino hits based on the wire-cell tagger
    kpalgo.set_input_larmatch_tree_name( hittree );
    kpalgo.set_verbosity( logger().level() );
    
    int kptypes[6] = {
      (int)larflow::kNuVertex,
      (int)larflow::kTrackStart,
      (int)larflow::kTrackEnd,
      (int)larflow::kShowerStart,
      (int)larflow::kShowerMichel,
      (int)larflow::kShowerDelta };
      
    int lfindex[6] = {
      17,
      18,
      19,
      20,
      21,
      22
    };
    
    for (int i=0; i<6; i++) {
      // (v2 larmatch-minkowski network neutrino-score index in hit)      
      kpalgo.set_keypoint_type( kptypes[i] );
      kpalgo.set_lfhit_score_index( lfindex[i] );      
      kpalgo.process( ioll );
    }
    
  }
}
}
