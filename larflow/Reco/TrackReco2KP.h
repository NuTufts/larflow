#ifndef __TRACK_RECO_2KP_H__
#define __TRACK_RECO_2KP_H__

#include <vector>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/track.h"

#include "KPCluster.h"

namespace larflow {
namespace reco {

  class TrackReco2KP : public larcv::larcv_base {
  public:

    TrackReco2KP()
      : larcv::larcv_base("TrackReco2KP"),
      _larflow_hit_treename("larmatch")
      {};
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
                                larlite::track& trackout,
                                larlite::larflowcluster& lfclusterout,
                                std::vector<int>& sp_used_v );

  protected:

    std::string _larflow_hit_treename;

  public:

    void set_larflow3dhit_tree_name( std::string name ) { _larflow_hit_treename=name; };
    


  protected:
    
    // collect the points
    struct Point_t {
      float s; // projection on line between start and endpt
      int idx; // index in lfhits array
      std::vector<float> pos; // 3d point
      bool operator<( const Point_t& rhs ) {
        if (s<rhs.s) return true;
        return false;
      };
    };

    void _prepareTrack( const std::vector<int>& trackpoints_v,
                        const std::vector<Point_t>& subset_v,
                        const larlite::event_larflow3dhit& lfhit_v,
                        const std::vector<float>& start,
                        const std::vector<float>& end,
                        larlite::track& track,
                        larlite::larflowcluster& lfcluster );
    
    
    
  };
  
}
}

#endif
