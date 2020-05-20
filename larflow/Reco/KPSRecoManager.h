#ifndef __KPS_RECO_MANAGER_H__
#define __KPS_RECO_MANAGER_H__

// larlite
#include "DataFormat/storage_manager.h"

// larcv
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

// ublarcvapp
#include "ublarcvapp/UBImageMod/EmptyChannelAlgo.h"

// larflow
#include "KeypointReco.h"
#include "SplitHitsBySSNet.h"
#include "DBScanLArMatchHits.h"
#include "TrackReco2KP.h"

namespace larflow {
namespace reco {
    
  class KPSRecoManager : public larcv::larcv_base {
  public:

    KPSRecoManager();
    virtual ~KPSRecoManager();

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    

    // image mods
    ublarcvapp::EmptyChannelAlgo _badchmaker;

    // keypoint reconstruction
    KeypointReco     _kpreco;
    SplitHitsBySSNet _splithits;
    TrackReco2KP     _tracker2kp;
    DBScanLArMatchHits _cluster_track;
    DBScanLArMatchHits _cluster_shower;    

    // Algorithms
    void recoParticles( larcv::IOManager& iolcv, larlite::storage_manager& ioll,
                        const std::vector<KPCluster>& kpcluster_v );
    
    
  };

}
}

#endif
