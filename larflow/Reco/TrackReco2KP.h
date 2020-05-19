#ifndef __TRACK_RECO_2KP_H__
#define __TRACK_RECO_2KP_H__

#include <vector>

#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"

#include "KPCluster.h"

namespace larflow {
namespace reco {

  class TrackReco2KP {
  public:

    TrackReco2KP() {};
    virtual ~TrackReco2KP() {};

    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  const std::vector<KPCluster>& kpcluster_v );

    struct KPInfo_t {
      int idx;
      int boundary_type;
      float dwall;
      bool operator< (const KPInfo_t& rhs ) {
        if ( dwall<rhs.dwall ) return true;
        return false;
      };
    };//< used to sort Keypoints
    
    struct KPPair_t {
      int start_idx;
      int end_idx;
      float dist2axis;
      float dist2pts;
      bool operator<( const KPPair_t& rhs ) {
        if ( dist2axis<rhs.dist2axis )
          return true;
        return false;
      };
    };//< used to sort pairs

    std::vector<int> makeTrack( const std::vector<float>& startpt,
                                const std::vector<float>& endpt,
                                const larlite::event_larflow3dhit& lfhits,
                                const std::vector<larcv::Image2D>& badch_v,                                
                                std::vector<int>& sp_used_v );
    

  };
  
}
}

#endif
