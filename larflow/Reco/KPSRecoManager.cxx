#include "KPSRecoManager.h"

namespace larflow {
namespace reco {

  KPSRecoManager::KPSRecoManager()
  {    
    _tracker2kp.set_larflow3dhit_tree_name( "showerhit" ); // take output of splitter
  }

  KPSRecoManager::~KPSRecoManager()
  {
  }
  
  void KPSRecoManager::process( larcv::IOManager& iolcv,
                                larlite::storage_manager& ioll )
  {

    // load the data we need
    _splithits.process( iolcv, ioll );

    // KEYPOINT RECO: make keypoint candidates
    _kpreco.process( ioll );

    // PARTICLE RECO
    recoParticles( iolcv, ioll, _kpreco.output_pt_v );
    

    // INTERACTION RECO

    // Cosmic reco: tracks at end points + deltas and michels

    // Multi-prong internal reco

    // Single particle interactions
    
  }

  void KPSRecoManager::recoParticles( larcv::IOManager& iolcv,
                                      larlite::storage_manager& ioll,
                                      const std::vector<KPCluster>& kpcluster_v )
  {

    // TRACK 2-KP RECO: make tracks using pairs of keypoints
    _tracker2kp.process( iolcv, ioll, kpcluster_v );

    // TRACK 1-KP RECO: make tracks using clusters and single keypoint

    // SHOWER 1-KP RECO: make shower using clusters and single keypoint

    // TRACK CLUSTER-ONLY RECO: make tracks without use of keypoints

    // SHOWER CLUSTER-ONLY RECO: make showers without use of keypoints
    
  }
  
}
}
