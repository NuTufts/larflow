#ifndef __LARFLOW_RECO_TRACK_CLUSTER_BUILDER_H__
#define __LARFLOW_RECO_TRACK_CLUSTER_BUILDER_H__

#include <map>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "DataFormat/storage_manager.h"
#include "DataFormat/pcaxis.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/track.h"

namespace larflow {
namespace reco {

  class TrackClusterBuilder : public larcv::larcv_base {

  public:

    TrackClusterBuilder()
      : larcv::larcv_base("TrackClusterBuilder"),
      _one_track_per_startpoint(true)
        {};
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
      float   cosine_seg; //< cosine between two segment directions
      std::vector<float> dir;
    };

    struct NodePos_t {
      int nodeidx;
      int segidx;
      bool inpath;
      std::vector<float> pos;
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
      float len;
      int idx;
      bool inpath;
      bool visited;

      Segment_t()
      : len(0),
        idx(-1),
        inpath(false),
        visited(false)
      {};
      Segment_t( std::vector<float>& s, std::vector<float>& e ) {
        //if ( s[1]<e[1] ) {
        start = s;
        end = e;
        /* } */
        /* else  { */
        /*   start = e; */
        /*   end = s; */
        /* } */
        len = 0.;
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
        inpath = false;
      };
    };

    /* struct Path_t { */
    /*   std::vector<Segment_t*> seg_v; */
    /*   std::vector<float>      segdir_v; */
    /*   float pathlen; */
    /*   float seglen; */
    /* }; */

  protected:

    // DATA CONTAINERS
    // ----------------
    
    // library of all segments created
    std::vector< Segment_t > _segment_v;
    std::vector< NodePos_t > _nodepos_v;

    // connection libraries
    // two types:
    //  (1) connection between ends of the same segment
    //  (2) connection between ends of different segments
    std::map< std::pair<int,int>, Connection_t > _connect_m; // index is directional, index1->index2
    std::map< std::pair<int,int>, Connection_t > _segedge_m; // index is directional, index1->index2    

    // track proposals
    std::vector< std::vector<NodePos_t*> > _track_proposal_v;

    // PARAMETERS
    // -----------
    bool _one_track_per_startpoint;

  public:
    
    void loadClusterLibrary( const larlite::event_larflowcluster& cluster_v,
                             const larlite::event_pcaxis& pcaxis_v );

    void buildConnections();
    void buildNodeConnections();

    void buildTracksFromPoint( const std::vector<float>& startpoint );

    void clear();
    
    void clearProposals() { _track_proposal_v.clear(); };

    void fillLarliteTrackContainer( larlite::event_track& ev_track );

    void set_output_one_track_per_startpoint( bool output_only_one ) { _one_track_per_startpoint=output_only_one; };

    std::string str( const Segment_t& seg );
    
  protected:

    void _recursiveFollowPath( NodePos_t& node,
                               std::vector<float>& path_dir,
                               std::vector<NodePos_t*>& path,
                               std::vector< const std::vector<float>* > path_dir_v,
                               std::vector< std::vector<NodePos_t*> >& complete );
    
    
    void _choose_best_paths( std::vector< std::vector<NodePos_t*> >& complete_v, 
                             std::vector< std::vector<NodePos_t*> >& filtered_v );
    
  };
  
}
}


#endif
