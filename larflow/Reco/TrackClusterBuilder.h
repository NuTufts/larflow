#ifndef __LARFLOW_RECO_TRACK_CLUSTER_BUILDER_H__
#define __LARFLOW_RECO_TRACK_CLUSTER_BUILDER_H__

#include <map>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"

#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/pcaxis.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/track.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class TrackClusterBuilder
   * @brief Make tracks by associating clusters based on principle components
   *
   * We approach this as a graph traveral problem.
   * We use the first principle component of the clusters to define a line segment.
   * The ends of the line segments serve as our nodes in the graph.
   * We have two types of edges in the graph
   * @verbatim embed:rst:leading-asterisk
   *  * segment edges (stored in `_segedge_m`): that connect the end points on the same line segment
   *  * connection edges (stored in `_connect_m`): that connect end points on different line segments
   * @endverbatim
   *
   * We use depth-first traveral to define paths from a given segment end point, to 
   * all connected segments. When we traverse onto a new segment via a connection edge,
   * we then must travel to the end point on the same segment via a segment edge.
   * The method performing the traversal is the recursive function, _recursiveFollowPath().
   *
   * Traversal stops when we hit a segment endpoint with no connection edges.
   *
   * Also, when we traverse across a connection edge, we only do so under certain conditions.
   * These conditions are based on the length of the edge (i.e. distance between the segment end points)
   * and the relative direction of the segments.
   *
   */
  class TrackClusterBuilder : public larcv::larcv_base {

  public:

    TrackClusterBuilder()
      : larcv::larcv_base("TrackClusterBuilder"),
	_max_node_endpt_dist(100.0),
	_one_track_per_startpoint(true)
        {};
    virtual ~TrackClusterBuilder() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );


    struct Segment_t;
    
    /**
     * @struct Connection_t
     * @brief represents a connection (edge) between segments (node)
     */
    struct Connection_t {
      Segment_t* node;        ///< pointer to segment
      int     from_seg_idx;   ///< from node
      int     to_seg_idx;     ///< to node
      int     endidx;         ///< end of the segment we are connected to
      float   dist;           ///< distance between ends, i.e. length of connection
      float   cosine;         ///< cosine between segment direction and endpt-connections
      float   cosine_seg;     ///< cosine between two segment directions
      std::vector<float> dir; ///< direction of connection 
    };

    /**
     * @struct NodePos_t
     * @brief represents a segment end point. serves as a node.
     */
    struct NodePos_t {
      int nodeidx;            ///< node index
      int segidx;             ///< segment index
      bool inpath;            ///< flag indicating it is currently part of a path
      bool veto;              ///< flag that if true, we skip this node when connecting paths
      std::vector<float> pos; ///< position of the node (segment end)
      NodePos_t()
      : nodeidx(-1),
        segidx(-1),
        inpath(false),
        veto(false),
        pos( {0,0,0} )
      {};
    };
    
    // This represents a line segment
    /**
     * @struct Segment_t
     * @brief represents a cluster through a line segment made from the first principle component
     */
    struct Segment_t {
      const larlite::larflowcluster* cluster;  ///< pointer to cluster this segment derives from
      const larlite::pcaxis*         pca;      ///< pointer to principle component for the cluster instance
      const larlite::track*          trackseg; ///< pointer to track object containing fitted track segment path
      std::vector<float> start;                ///< start of the line segment
      std::vector<float> end;                  ///< end of the line segment
      std::vector<float> dir;                  ///< direction from start to end
      std::vector< Connection_t* > con_start;  ///< pointer to Connection_t (edges) connected to the segment start point
      std::vector< Connection_t* > con_end;    ///< pointer to Connection_t (edges) connected to the segment end point
      float len;                               ///< length of the line segment
      int idx;                                 ///< index of the segment
      bool inpath;                             ///< flag indicating segment is currently part of a path
      bool visited;                            ///< flag indicating segment has been visited at least once

      Segment_t()
      : len(0),
        idx(-1),
        inpath(false),
        visited(false)
      {};

      /**
       * @brief constructor with start and end points of segment 
       * @param[in] s start point of segment
       * @param[in] e end point of segment
       */
      Segment_t( std::vector<float>& s, std::vector<float>& e ) {
        start = s;
        end = e;
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

  protected:

    // DATA CONTAINERS
    // ----------------
    
    // library of all segments created
    std::vector< Segment_t > _segment_v;  ///< container of segments made from larflowcluster instances
    std::vector< NodePos_t > _nodepos_v;  ///< container of end points serving as nodes in our graph

    // connection libraries
    // two types:
    //  (1) connection between ends of the same segment
    //  (2) connection between ends of different segments
    std::map< std::pair<int,int>, Connection_t > _connect_m; ///< map from _nodepos_v index to connection edge that connect end points on different segments; index is directional, index1->index2
    std::map< std::pair<int,int>, Connection_t > _segedge_m; ///< map from _nodepos_v indices to segment edge that connect end points on the same segment; index is directional, index1->index2    

    // track proposals
    std::vector< std::vector<NodePos_t*> > _track_proposal_v; ///< sequence of NodePos_t pointers, representing a track or path

    // PARAMETERS
    // -----------
    float _max_node_endpt_dist;  ///< maximum distance two cluster ends can be connected by an edge
    bool _one_track_per_startpoint; ///< if flag is true, reduce many possible paths down to one

  public:
    
    void loadClusterLibrary( const larlite::event_larflowcluster& cluster_v,
                             const larlite::event_pcaxis& pcaxis_v,
                             const larlite::event_track& trackseg_v );                             

    void buildNodeConnections( const std::vector<larcv::Image2D>* padc_v=nullptr,
                               const std::vector<larcv::Image2D>* pbadch_v=nullptr);

    void buildTracksFromPoint( const std::vector<float>& startpoint );

    void clear();
    void clear_cluster_data();
    void clear_track_proposals();

    /** @brief clear track proposal container _track_proposal_v */
    void clearProposals() { _track_proposal_v.clear(); };

    void fillLarliteTrackContainer( larlite::event_track& ev_track,
                                    larlite::event_larflowcluster& evout_trackcluster );

    void fillLarliteTrackContainerWithFittedTrack( larlite::event_track& evout_track,
                                                   larlite::event_larflowcluster& evout_hitcluster,
                                                   const std::vector<larcv::Image2D>& adc_v );
    
    /** @brief set flag that if true, reduces many possible paths down to one */
    void set_output_one_track_per_startpoint( bool output_only_one ) { _one_track_per_startpoint=output_only_one; };

    std::string str( const Segment_t& seg );

    void resetVetoFlags();

    int findClosestSegment( const std::vector<float>& testpt, const float max_dist );
                            
    // tools to save details for debug/visualization
    void saveConnections( larlite::storage_manager& ioll, std::string tree_name="tcb_connections" );
    
  protected:

    void _recursiveFollowPath( NodePos_t& node,
                               std::vector<float>& path_dir,
                               std::vector<NodePos_t*>& path,
                               std::vector< const std::vector<float>* > path_dir_v,
                               std::vector< std::vector<NodePos_t*> >& complete );
    
    
    void _choose_best_paths( std::vector< std::vector<NodePos_t*> >& complete_v, 
                             std::vector< std::vector<NodePos_t*> >& filtered_v );

    void _buildTracksFromSegments();

    bool _checkForMissingVplane( const std::vector<float>& frac_v,
				 const std::vector<float>& seg_dir,
				 const float frac_threshold );
    
    
  };
  
}
}


#endif
