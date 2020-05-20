#ifndef __KPS_RECO_MANAGER_H__
#define __KPS_RECO_MANAGER_H__

// larlite
#include "DataFormat/storage_manager.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"

// larflow
#include "KeypointReco.h"
#include "SplitHitsBySSNet.h"
#include "TrackReco2KP.h"

namespace larflow {
namespace reco {
    
  class KPSRecoManager {
  public:

    KPSRecoManager();
    virtual ~KPSRecoManager();

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    


    // keypoint reconstruction
    KeypointReco     _kpreco;
    SplitHitsBySSNet _splithits;
    TrackReco2KP     _tracker2kp;


    // Algorithms
    void recoParticles( larcv::IOManager& iolcv, larlite::storage_manager& ioll,
                        const std::vector<KPCluster>& kpcluster_v );
    
    
  };

}
}

#endif
