#ifndef __LARFLOW_RECO_TRACK_CLUSTER_BUILDER_H__
#define __LARFLOW_RECO_TRACK_CLUSTER_BUILDER_H__

#include <map>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "DataFormat/storage_manager.h"
#include "DataFormat/pcaxis.h"
#include "DataFormat/larflowcluster.h"

namespace larflow {
namespace reco {

  class TrackClusterBuilder : public larcv::larcv_base {

  public:

    TrackClusterBuilder() {};
    virtual ~TrackClusterBuilder() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );


    struct Segment_t;
    
    // this represents a connection between segments
    struct Connection_t {
      Segment_t* node; //< pointer to segment
      int     from_seg_idx;
      int     to_seg_idx;
      int     endidx;  //< end of the segment we are connected to
      float   dist;    //< distance between ends, i.e. length of connection
      float   cosine;  //< cosine between segment direction and endpt-connections
    };
    
    // This represents a line segment
    struct Segment_t {
      const larlite::larflowcluster* cluster;
      const larlite::pcaxis*         pca;
      std::vector<float> start;
      std::vector<float> end;
      std::vector<float> dir;
      std::vector< Connection_t* > con_start;
      std::vector< Connection_t* > con_end;
      int idx;

      Segment_t() {};
      Segment_t( std::vector<float>& s, std::vector<float>& e ) {
        if ( s[1]<e[1] ) {
          start = s;
          end = e;
        }
        else  {
          start = e;
          end = s;
        }
        float len = 0.;
        dir.resize(3,0);
        for (int i=0; i<3; i++ ) {
          dir[i] = end[i]-start[i];
          len += dir[i]*dir[i];
        }
        len = sqrt(len);
        if ( len>0 ) {
          for (int i=0; i<3; i++)
            dir[i] /= len;
        }
        cluster = nullptr;
        pca = nullptr;
        idx = -1;
      };
    };

  protected:

    // library of all segments created
    std::vector< Segment_t > _segment_v;

    // connection library
    std::map< std::pair<int,int>, Connection_t > _connect_m; // index is directional, index1->index2

  public:
    
    void loadClusterLibrary( const larlite::event_larflowcluster& cluster_v,
                             const larlite::event_pcaxis& pcaxis_v );

    void buildConnections();
    
  };
  
}
}


#endif
