#include "TrackClusterBuilder.h"

#include "TVector3.h"
#include "larflow/Reco/geofuncs.h"

#include "ublarcvapp/ubdllee/dwall.h"
#include "larflow/Reco/ProjectionDefectSplitter.h"
#include "larflow/Reco/cluster_functions.h"

namespace larflow {
namespace reco {

  /**
   * @brief Build tracks using clusters in the event data managers 
   *   
   * @param[in] iolcv LArCV IOManager containing event data
   * @param[in] ioll  larlite storage_manager containing event data
   */
  void TrackClusterBuilder::process( larcv::IOManager& iolcv,
                                     larlite::storage_manager& ioll )
  {

    std::string producer = "trackprojsplit_full";
    
    larlite::event_larflowcluster* ev_cluster
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, producer);
    larlite::event_pcaxis* ev_pcaxis
      = (larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis,producer);
    larlite::event_track* ev_track_segments
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack,producer);

    loadClusterLibrary( *ev_cluster, *ev_pcaxis, *ev_track_segments );

  }

  /** @brief Clears internal event data containers */
  void TrackClusterBuilder::clear()
  {
    _segment_v.clear();
    _segedge_m.clear();
    _connect_m.clear();
    _nodepos_v.clear();
    _track_proposal_v.clear();
  }

  /**
   * @brief store given clusters and associated principle component info as Segment_t and NodePos_t
   *
   * The track segment object is optional.
   * If the container is empty, the pointer in Segment_t to the track segment
   * is set to nullptr.
   * 
   * @param[in] cluster_v  Vector of clusters in the form of larflowcluster 
   * @param[in] pcaxis_v   Principle component analysis info for the clusters in cluster_v
   * @param[in] trackseg_v List of tracks made by the line-segment fitter in larflow::reco::ProjectionDefectSplitter
   */
  void TrackClusterBuilder::loadClusterLibrary( const larlite::event_larflowcluster& cluster_v,
                                                const larlite::event_pcaxis& pcaxis_v,
                                                const larlite::event_track& trackseg_v )
  {

    if ( trackseg_v.size()>0 && cluster_v.size()!=trackseg_v.size() ) {
      std::stringstream sserr;
      sserr << "Number of track segments (" << trackseg_v.size() << ") does not match number of clusters (" << cluster_v.size() << ")" << std::endl;
      throw std::runtime_error( sserr.str() );
    }

    if ( pcaxis_v.size()>0 && cluster_v.size()!=pcaxis_v.size() ) {
      std::stringstream sserr;
      sserr << "Number of PC axes (" << pcaxis_v.size() << ") does not match number of clusters (" << cluster_v.size() << ")" << std::endl;
      throw std::runtime_error( sserr.str() );
    }
    
    for (int i=0; i<cluster_v.size(); i++) {
      
      const larlite::larflowcluster& cluster = cluster_v.at(i);
      const larlite::pcaxis& pca = pcaxis_v.at(i);

      const larlite::track* trackseg = nullptr;
      if ( trackseg_v.size()>0 )
        trackseg = &trackseg_v.at(i);

      // create a segment object
      std::vector<float> start(3,0);
      std::vector<float> end(3,0);
      for (int v=0; v<3; v++){
        start[v] = pca.getEigenVectors()[3][v];
        end[v]   = pca.getEigenVectors()[4][v];
      }

      Segment_t seg( start, end );
      seg.cluster  = &cluster;
      seg.pca      = &pca;
      seg.trackseg = trackseg;

      if ( seg.len<1.0 )
        continue;
      
      _segment_v.push_back(seg);
      _segment_v.back().idx = (int)_segment_v.size()-1;
      
      NodePos_t node1;
      node1.segidx  = _segment_v.back().idx;
      node1.nodeidx = (int)_nodepos_v.size();
      node1.pos     = start;

      NodePos_t node2;
      node2.segidx  = _segment_v.back().idx;
      node2.nodeidx = (int)_nodepos_v.size()+1;
      node2.pos     = end;

      _nodepos_v.push_back( node1 );
      _nodepos_v.push_back( node2 );

      // make segment edges
      Connection_t segedge1to2;
      segedge1to2.node = &_segment_v.back();
      segedge1to2.from_seg_idx = node1.nodeidx;
      segedge1to2.to_seg_idx   = node2.nodeidx;
      segedge1to2.dist         = seg.len;
      segedge1to2.cosine = 0.;
      segedge1to2.cosine_seg = 0.;
      segedge1to2.dir = seg.dir;

      Connection_t segedge2to1;
      segedge2to1.node = &_segment_v.back();
      segedge2to1.from_seg_idx = node2.nodeidx;
      segedge2to1.to_seg_idx   = node1.nodeidx;
      segedge2to1.dist         = seg.len;
      segedge2to1.cosine = 0.;
      segedge2to1.cosine_seg = 0.;
      segedge2to1.dir = seg.dir;
      for (int v=0; v<3; v++) segedge2to1.dir[v] *= -1.0;

      _segedge_m[ std::pair<int,int>( node1.nodeidx, node2.nodeidx ) ] = segedge1to2;
      _segedge_m[ std::pair<int,int>( node2.nodeidx, node1.nodeidx ) ] = segedge2to1;
      
      
    }

    LARCV_INFO() << "Stored " << _segment_v.size() << " track segments" << std::endl;
    LARCV_INFO() << "Stored " << _nodepos_v.size() << " segment endpoints as nodes" << std::endl;
  }

  /**
   * @brief build possible connections between end points of different segments
   *
   * nodes are connected as long as they are less than 100 cm from one another.
   * we also only allow a node (i.e. segment end point) to connect to only
   * one of the end points of another segment.
   *
   */
  void TrackClusterBuilder::buildNodeConnections()
  {

    for (int inode=0; inode<(int)_nodepos_v.size(); inode++ ) {
      NodePos_t& node_i = _nodepos_v[inode];

      for (int jnode=inode+1; jnode<(int)_nodepos_v.size(); jnode++) {

        // do not connect nodes on the same segment
        auto it_segedge = _segedge_m.find( std::pair<int,int>(inode,jnode) );
        if ( it_segedge!=_segedge_m.end() )
          continue;
                
        // define a connection in both directions
        NodePos_t& node_j = _nodepos_v[jnode];

        float dist = 0.;
        std::vector<float> dir_ij(3,0);
        for (int v=0; v<3; v++) {
          dir_ij[v] = node_j.pos[v]-node_i.pos[v];
          dist += dir_ij[v]*dir_ij[v];
        }
        dist = sqrt(dist);
        if ( dist>0 ) for (int v=0; v<3; v++) dir_ij[v] /= dist;

        if ( dist>100.0 ) continue;

        // make sure this is the shortest distance
        // between the segments
        int jnode_pair = -1;
        if ( jnode%2==0 ) {
          jnode_pair = jnode+1;
        }
        else
          jnode_pair = jnode-1;

        NodePos_t& node_j_pair = _nodepos_v[jnode_pair];
        float pairdist = 0.;
        for (int v=0; v<3; v++ ) {
          pairdist += ( node_j_pair.pos[v]-node_i.pos[v] )*( node_j_pair.pos[v]-node_i.pos[v] );
        }
        pairdist = sqrt(pairdist);

        if ( pairdist<dist ) {
          // the node's segment pair is closer to node_i, so we do not make this connection
          continue;
        }
          
        // i->j
        Connection_t con_ij;
        con_ij.node = nullptr;
        con_ij.from_seg_idx = inode;
        con_ij.to_seg_idx   = jnode;
        con_ij.dist = dist;
        con_ij.endidx = 0;
        con_ij.cosine = 0.;
        con_ij.cosine_seg = 0.;        
        con_ij.dir = dir_ij;
        
        Connection_t con_ji;
        con_ji.node = nullptr;
        con_ji.from_seg_idx = jnode;
        con_ji.to_seg_idx   = inode;
        con_ji.dist   = dist;
        con_ij.endidx = 0;
        con_ji.cosine = 0.;
        con_ji.cosine_seg = 0.;
        con_ji.dir = dir_ij;
        for (int v=0; v<3; v++) {
          con_ji.dir[v] *= -1.0;
        }

        _connect_m[ std::pair<int,int>(inode,jnode) ] = con_ij;
        _connect_m[ std::pair<int,int>(jnode,inode) ] = con_ji;
        
      }      
    }

    LARCV_INFO() << "Made " << _connect_m.size() << " nodepos connections" << std::endl;
  }
  
  /**
   * @brief Seed the tracking algorithm with a startig point
   *
   * The initial point will be used to find the closest line segment.
   * The keypoint must be within 3 cm of the line segment, else no track(s)
   * will be made.
   * 
   * From there, the depth-first traversal is run twice, once for each
   * end point of the segment (unless one of the end points are veto'd).
   *
   * Resulting track(s) are stored in _track_proposal_v.
   *
   * @param[in] startpoint Position in 3D to find segment that will seed track(s).
   *
   */
  void TrackClusterBuilder::buildTracksFromPoint( const std::vector<float>& startpoint )
  {

    // find closest segment
    float max_dist = 3.0;
    int   min_segidx = findClosestSegment( startpoint, max_dist );
    
    if ( min_segidx<0 ) {
      LARCV_INFO() << "No acceptable segment found for startpoint" << std::endl;
      return;
    }

    LARCV_DEBUG() << "=== Seed track with point (" << startpoint[0] << "," << startpoint[1] << "," << startpoint[2] << ") ======" << std::endl;
    LARCV_DEBUG() << " segment found idx=[" << min_segidx << "] to build from " << str(_segment_v[min_segidx]) << std::endl;
    
    // reset the nodes
    for ( auto& node : _nodepos_v )
      node.inpath = false;

    // Nodes from the segment
    NodePos_t& node1 = _nodepos_v[2*min_segidx];
    NodePos_t& node2 = _nodepos_v[2*min_segidx+1];
    node1.inpath = true;
    node2.inpath = true;

    NodePos_t* startnode = nullptr;
    NodePos_t* nextnode  = nullptr;
    float dist1 = 0.;
    float dist2 = 0.;
    for (int v=0; v<3; v++) {
      dist1 += (node1.pos[v]-startpoint[v])*(node1.pos[v]-startpoint[v]);
      dist2 += (node2.pos[v]-startpoint[v])*(node2.pos[v]-startpoint[v]);
    }
    startnode = (dist1<dist2) ? &node1 : &node2;
    nextnode  = (dist1>dist2) ? &node1 : &node2;

    LARCV_DEBUG() << " starting track from: (" << startnode->pos[0] << "," << startnode->pos[1] << "," << startnode->pos[2] << ")" << std::endl;
    LARCV_DEBUG() << " first gap edge from: (" << nextnode->pos[0] << "," << nextnode->pos[1] << "," << nextnode->pos[2] << ")" << std::endl;

    auto it_segedge12 = _segedge_m.find( std::pair<int,int>(startnode->nodeidx,nextnode->nodeidx) );
    auto it_segedge21 = _segedge_m.find( std::pair<int,int>(nextnode->nodeidx,startnode->nodeidx) );    
    std::vector<float>& path_dir = it_segedge12->second.dir;

    std::vector< NodePos_t* > path;
    std::vector< const std::vector<float>* > path_dir_v;    
    std::vector< std::vector<NodePos_t*> > complete_v;

    // start to nextnode
    if ( !startnode->veto ) {
      path.clear();
      path_dir_v.clear();      
      path.push_back( startnode );
      path_dir_v.push_back( &path_dir );    
      _recursiveFollowPath( *nextnode, path_dir, path, path_dir_v, complete_v );
      LARCV_DEBUG() << "[after start->next] point generated " << complete_v.size() << " possible tracks" << std::endl;
    }

    // flip things
    if ( !nextnode->veto ) {
      path.clear();
      path_dir_v.clear();
      path.push_back( nextnode );
      path_dir_v.push_back( &it_segedge21->second.dir );
      _recursiveFollowPath( *startnode, path_dir, path, path_dir_v, complete_v );
      LARCV_DEBUG() << "[after next->start] point generated " << complete_v.size() << " possible tracks" << std::endl;
    }
    

    // this will generate proposals. we must choose among them
    std::vector< std::vector<NodePos_t*> > filtered_v;    
    _choose_best_paths( complete_v, filtered_v );

    for ( auto& path : filtered_v )  {
      LARCV_DEBUG() << "storing path nnodes=" << path.size()
                    << " node[" << path.front()->nodeidx << "]->node[" << path.back()->nodeidx << "]"
                    << " seg[" << path.front()->segidx << "]->seg[" << path.back()->segidx << "]"
                    << std::endl;
      _track_proposal_v.push_back( path );
    }

    LARCV_INFO() << "Number of paths now stored: " << _track_proposal_v.size() << std::endl;
    
  }  

  /**
   * @brief recursive function the implements depth-first graph traversal
   *
   * For given node, we check possible nodes (not on the same segment) to connect to.
   * Nodes are connected based on 
   * @verbatim embed:rst:leading-asterisk
   *  * distance between nodes
   *  * direction between the nodes
   *  * direction between the segments the two nodes sit on
   * @endverbatim
   *
   * If two nodes on different segments are connected, 
   * we automatically make an additional traversal to the other end of the new segment.
   *
   * Descent stops on leaf nodes, defined as nodes (i.e. end points on line segments)
   * with no valid connections.
   * 
   * When a leaf node is reached, we save the path of nodes to the leaf node
   * and go up one level in the graph.
   *
   * We enforce a maximum of 1000 possible completed paths to leaf nodes.
   * 
   * @param[in] node Current Node of path
   * @param[in] path_dir Current direction of path
   * @param[in] path  Sequence of node pointers along path
   * @param[in] path_dir_v  Sequence of directions along all nodes on path
   * @param[in] complete Collection of paths that have reached leaf nodes
   */
  void TrackClusterBuilder::_recursiveFollowPath( NodePos_t& node,
                                                  std::vector<float>& path_dir,
                                                  std::vector<NodePos_t*>& path,
                                                  std::vector< const std::vector<float>* > path_dir_v,
                                                  std::vector< std::vector<NodePos_t*> >& complete )
  {

    // we add ourselves to the current path
    node.inpath = true;
    path.push_back( &node );
    path_dir_v.push_back( &path_dir );
    
    // oh wow. we loop through possible connections, and descend.
    int nconnections = 0;
    float mindist = 1e9;
    float maxcos = 0;
    for (int inode=0; inode<_nodepos_v.size(); inode++) {

      NodePos_t& nextnode = _nodepos_v[inode];
      if ( nextnode.inpath || nextnode.veto ) continue;      

      auto it = _connect_m.find( std::pair<int,int>(node.nodeidx,inode) );
      if ( it==_connect_m.end() ) continue; // no connection

      //std::cout << "  (connect " << seg.idx << "->" << iseg << ") dist=" << it->second.dist << " cos="  << it->second.cosine << std::endl;

      // we connect based on:

      // distance between node (edge length)
      // direction between nodes (edge direction)
      // direction between nextnode and its pair (segment direction)
      int inode_pair = -1;
      if ( inode%2==0 )
        inode_pair = inode+1;
      else
        inode_pair = inode-1;

      auto it_segedge = _segedge_m.find( std::pair<int,int>(inode,inode_pair) );
      float& gaplen = it->second.dist;
      std::vector<float>&  gapdir = it->second.dir; // should have worked ...
      // std::vector<float>  gapdir(3,0);// = it->second.dir;
      // for (int v=0; v<3; v++) {
      //   gapdir[v] = (nextnode.pos[v]-node.pos[v])/gaplen;
      // }
      std::vector<float>& segdir = it_segedge->second.dir;

      float segcos = 0.;  // cosine between last segment and proposed segment
      float concos1 = 0.; // cosine between last segment and gap edge
      float concos2 = 0.; // cosine between gap edge and proposed segment dir
      for ( int v=0; v<3; v++ ) {
        segcos  += path_dir[v]*segdir[v];
        concos1 += path_dir[v]*gapdir[v];
        concos2 += gapdir[v]*segdir[v];
      }

      // add this segment
      // LARCV_DEBUG() << "  consider node[" << node.nodeidx << "]->node[" << nextnode.nodeidx << "] "
      //               << " seg[" << node.segidx << "]->seg[" << nextnode.segidx << "] "
      //               << "dist=" << gaplen << " "
      //               << " seg1-seg2=" << segcos
      //               << " seg1-edge=" << concos1
      //               << " edge-seg2=" << concos2
      //               << std::endl;
      
      // criteria for accepting connection
      // ugh, the heuristics ...
      if ( (gaplen<2.0 && segcos>0 ) // close: only check direction between segments
           || (gaplen>=2.0 && gaplen<10.0 && segcos>0.8 && concos1>0.8 && concos2>0.8 )  // far: everything in the same direction           
           || (gaplen>=10.0 && segcos>0.9 && concos1>0.9 && concos2>0.9 ) ) {  // far: everything in the same direction

        //std::cin.get();
        
        NodePos_t& node_pair = _nodepos_v[inode_pair];
        nextnode.inpath = true;
        path_dir_v.push_back( &segdir );
        path.push_back( &nextnode );
        nconnections++;

        // add this segment
        LARCV_DEBUG() << "  ==> connected node[" << node.nodeidx << "]->node[" << nextnode.nodeidx << "]->node[" << node_pair.nodeidx << "]" << std::endl;
        LARCV_DEBUG() << "    from: seg[" << node.segidx << "] (" << node.pos[0] << "," << node.pos[1] << "," << node.pos[2] << ")" << std::endl;
        LARCV_DEBUG() << "    to: seg[" << nextnode.segidx << "] (" << nextnode.pos[0] << "," << nextnode.pos[1] << "," << nextnode.pos[2] << ")"
                      << "--> (" << node_pair.pos[0] << "," << node_pair.pos[1] << "," << node_pair.pos[2] << ")" << std::endl;
        
        _recursiveFollowPath( node_pair, segdir, path, path_dir_v, complete );
        
        LARCV_DEBUG() << "--- Back to node[" << node.nodeidx << "] ---" << std::endl;

        if ( complete.size()>=1000 ) {
          //cut this off!
          std::cout << "  cut off search. track limit reached: " << complete.size() << std::endl;
          break;
        }
      }
    }
    
    if ( nconnections==0 && complete.size()<10000 && path.size()>1 ) {
      // was a leaf, so make a complete track.
      // else was a link
      complete.push_back( path );
      LARCV_DEBUG() << "reached a leaf, copy track len=" << path.size() << " to completed list. num of completed tracks: " << complete.size() << std::endl;      
    }
    //std::cout << "  mindist=" << mindist << " maxcos=" << maxcos << std::endl;
    // pop  off the last two nodes
    int nnodes = (int)path.size();
    if ( path.size()<2 )
      return;
    path[nnodes-2]->inpath = false;
    path[nnodes-1]->inpath = false;
    path.pop_back();
    path.pop_back();
    path_dir.pop_back();
    path_dir.pop_back();    

    return;
  }

  /**
   * @brief print segment info to standatd out
   *
   * @param[in] seg Segment whose info. will be printed
   */
  std::string TrackClusterBuilder::str( const TrackClusterBuilder::Segment_t& seg )
  {
    std::stringstream ss;
    ss << "[segment[" << seg.idx << "] len=" << seg.len;
    if ( seg.start.size()==3 )      
      ss << " start=(" << seg.start[0] << "," << seg.start[1] << "," << seg.start[2] << ") ";
    if (seg.end.size()==3 )
      ss << " end=(" << seg.end[0] << "," << seg.end[1] << "," << seg.end[2] << ") ";
    if (seg.dir.size()==3 )
      ss << " dir=(" << seg.dir[0] << "," << seg.dir[1] << "," << seg.dir[2] << ") ";
    ss << "]";
    return ss.str();
  }

  /**
   * @brief reset veto flags in nodes
   */
  void TrackClusterBuilder::resetVetoFlags()
  {
    for ( auto& node : _nodepos_v ) {
      node.veto = false;
    }
  }

  /**
   * @brief choose the best path among the several created after the recursive graph traversal is finished
   *
   * from a given start point, several completed paths to various leaf nodes are possible and returned.
   * from these we want to pick the "best" path from the start point to one or more of the leaf nodes.
   *
   * we narrow down in phases
   * @verbatim embed:rst:leading-asterisk
   *   * phase 1: we group completed paths by leaf segment, choosing min-dist path + paths within 20 cm of min dist
   *   * phase 2: we choose one path among the leaf group, by the highest fraction of the path length is from cluster segment lengths
   *              in other words the path with the least amount of gap connections.
   *   * phase 3: we choose the path with the longest length if _one_track_per_startpoint is true.
   * @endverbatim
   *
   * @param[in] complete_v A vector containing all paths to leaf nodes from a single start point
   * @param[out] filtered_v A vector where we store all selected paths, based on several criteria
   */
  void TrackClusterBuilder::_choose_best_paths( std::vector< std::vector<NodePos_t*> >& complete_v,
                                                std::vector< std::vector<NodePos_t*> >& filtered_v )
  {


    typedef std::vector<NodePos_t*> Path_t;
    
    // phase 1, group paths by leaf index (the index of the last node)
    std::map<int, std::vector< Path_t* > > leaf_groups;
    for ( auto& path : complete_v ) {
      int leafidx = path.back()->nodeidx;
      auto it = leaf_groups.find(leafidx);
      if ( it==leaf_groups.end() ) {
        leaf_groups[leafidx] = std::vector< Path_t* >();
        it = leaf_groups.find(leafidx);
      }
      it->second.push_back( &path );
    }

    // loop over leaf groups, choosing subset close to min
    int nfiltered_before = (int)filtered_v.size();
    std::vector< Path_t* > selected_v;
    std::vector<float> path_dist_ratio_v;
    std::vector<float> selected_pathlen_v;
    std::vector<float> selected_enddist_v;
    float min_pathdist_ratio = 1.0e9;
    path_dist_ratio_v.reserve( leaf_groups.size() );
    for ( auto it=leaf_groups.begin(); it!=leaf_groups.end(); it++ ) {

      //std::cout << "=== Find mindist subgroup in Leaf Group[" << it->first << "] ===" << std::endl;

      std::vector< float > pathlen_v;
      std::vector< float > seglen_v;
      pathlen_v.reserve(it->second.size());
      seglen_v.reserve(it->second.size());

      int minpath_idx = -1;
      float minpathlen = 1e9;
      
      for ( size_t ipath=0; ipath<it->second.size(); ipath++ ) {
        Path_t* ppath = it->second.at(ipath);
        float pathdist = 0.;
        float totseglen = 0.;
        for ( size_t inode=1; inode<ppath->size(); inode++ ) {

          NodePos_t* pnode = ppath->at(inode);
          NodePos_t* pnode_prev = ppath->at(inode-1);

          float dist = 0.;
          for (int v=0; v<3; v++)
            dist += ( pnode->pos[v]-pnode_prev->pos[v] )*( pnode->pos[v]-pnode_prev->pos[v] );
          dist = sqrt(dist);

          if ( inode%2==1 )
            totseglen += dist;
          pathdist += dist;
          
        }//end of loop over path segments
        //std::cout << "  Leaf-group[" << it->first << "] path[" << ipath << "] pathlen=" << pathdist << " seglen=" << totseglen << std::endl;
        pathlen_v.push_back(pathdist);
        seglen_v.push_back(totseglen);
        if ( pathdist<minpathlen ) {
          minpath_idx = (int)ipath;
          minpathlen = pathdist;
        }
      }//end of path loop
      //std::cout << " Leaf group minpathlen=" << minpathlen << std::endl;
      
      // get max seg/path frac out of paths within minimum dist
      int max_segfrac_idx = -1;
      float max_segfrac = 0.;
      for (size_t ipath=0; ipath<pathlen_v.size(); ipath++ ) {
        if ( fabs(pathlen_v[ipath]-minpathlen) < 20.0 ) {
          float segfrac = seglen_v[ipath]/pathlen_v[ipath];
          if ( segfrac>max_segfrac ) {
            max_segfrac = segfrac;
            max_segfrac_idx = ipath;
          }
        }
      }
      //std::cout << " Leaf sub-group maxsegfrac=" << max_segfrac <<" pathidx=" << max_segfrac_idx << std::endl;
      selected_v.push_back( it->second.at(max_segfrac_idx) );
      selected_pathlen_v.push_back( pathlen_v[max_segfrac_idx] );
      
      // store path/dist ratio
      Path_t* leafpath = it->second.at(max_segfrac_idx);
      

      float end_dist = 0.;
      for (int v=0; v<3; v++) {
        float dx = leafpath->front()->pos[v] - leafpath->back()->pos[v];
        end_dist += dx*dx;
      }
      end_dist = sqrt(end_dist);
      selected_enddist_v.push_back( end_dist );      
      
      float pathdist_ratio = 0.;
      if ( end_dist>0 ) {
        pathdist_ratio = pathlen_v[max_segfrac_idx]/end_dist;
      }
      path_dist_ratio_v.push_back( pathdist_ratio );


      if ( pathdist_ratio>0 && pathdist_ratio<min_pathdist_ratio ) {
        min_pathdist_ratio = pathdist_ratio;
      }
      
      //std::cout << "  Leafgroup["  << it->first << "] best path dist between ends: " << end_dist << std::endl;
      
      //std::cout << "=== End of Leafgroup[" << it->first << "] selection path/dist ratio: " << pathdist_ratio << " len=" << pathlen_v[max_segfrac_idx] << " ===" << std::endl;
    }//end of group loop

    // so far we've chosen best path from seed point to leaf point,
    // now choose among leaf points, if _one_track_per_startpoint is true.
    
    if ( _one_track_per_startpoint ) {
      // choose among the best of the best?
      // the minimum ratio of path-length to 
      float min_track_path_dist_ratio = 1e9;
      float max_track_dist = 0.;
      int min_max_len_idx = -1;
      for ( size_t i=0; i<selected_v.size(); i++ ) {
        if ( path_dist_ratio_v[i]>=1.0 && path_dist_ratio_v[i]<min_pathdist_ratio+0.2 ) {
          // if ( min_track_path_dist_ratio>path_dist_ratio_v[i] ) {
          //   min_max_len_idx = i;
          //   min_track_path_dist_ratio = path_dist_ratio_v[i];
          // }
          if ( max_track_dist<selected_enddist_v[i] ) {
            max_track_dist = selected_enddist_v[i];
            min_max_len_idx = i;
        }
        }
      }
      
      if ( min_max_len_idx>=0 ) {
        filtered_v.push_back( *selected_v[min_max_len_idx] );
      }
    }
    else {
      // allow us to return a collection of paths from the same start point to many different leaves
      for ( auto& ppath : selected_v )
        filtered_v.push_back ( *ppath );
      std::cout << "Number of leaf-group candidate tracks return: " << (int)filtered_v.size()-nfiltered_before << std::endl;          
    }

  }

  /**
   * @brief store paths we've found as larlite::track objects in the provided event container
   *
   * We must convert a track defined as a sequence of NodePos_t instances into
   * a larlite track for storage into the output larlite ROOT file.
   *
   * This builds a track using the pca-axes.
   *
   * @param[out] evout_track Node paths stored in _track_proposal_v are stored into this container
   */
  void TrackClusterBuilder::fillLarliteTrackContainer( larlite::event_track& evout_track )
  {

    for ( size_t itrack=0; itrack<_track_proposal_v.size(); itrack++ ) {

      auto const& path = _track_proposal_v[itrack];

      if ( path.size()<=1 )
        continue;
      
      larlite::track lltrack;
      lltrack.reserve(path.size()); // npts = 2*num seg;

      std::vector<float> last_seg_dir(3,0);
      for (int inode=1; inode<path.size(); inode++ ) {

        const NodePos_t* node     = path[inode];
        const NodePos_t* nodeprev = path[inode-1];
        float d = 0.;
        for (int v=0; v<3; v++) {
          last_seg_dir[v] = node->pos[v]-nodeprev->pos[v];
          d += last_seg_dir[v]*last_seg_dir[v];
        }
        d = sqrt(d);
        if ( d>0 ) {
          for (int v=0; v<3; v++)
            last_seg_dir[v] /= d;
        }
        
        lltrack.add_vertex( TVector3(nodeprev->pos[0],nodeprev->pos[1], nodeprev->pos[2]) );
        lltrack.add_direction( TVector3(last_seg_dir[0],last_seg_dir[1], last_seg_dir[2]) );
        
        if ( inode+1==(int)path.size() ) {
          lltrack.add_vertex( TVector3(node->pos[0],node->pos[1], node->pos[2]) );
          lltrack.add_direction( TVector3(last_seg_dir[0],last_seg_dir[1], last_seg_dir[2]) );
        }
        
      }//end of loop over nodes

      evout_track.emplace_back( std::move(lltrack) );
    }//end of loop over tracks
    
  }

  /**
   *
   * @brief Make tracks from the stored collection of cluster line segments
   *
   * Intended to be used after making tracks from keypoints.
   * Mark existing segments in paths as "used".
   *
   * Tracks are seeded with segments based on their distance to the edge of the TPC walls.
   * The walls tested are the upstream, downstream, top and bottom.
   * The distance to the anode and cathode are not used.
   *
   * The algorithm is greedy in that, once a segment is include in a path, the
   * segment is not allowed to a part of any other path.
   * 
   * Possible segments to seed the tracks are from _segment_v.
   *
   */
  void TrackClusterBuilder::_buildTracksFromSegments()
  {

    // mark segments used
    int unused = 0;
    std::vector<int> used_v( _segment_v.size(), 0 );
    for ( auto const& path : _track_proposal_v ) {
      for ( auto const& pnode : path ) {
        used_v[ pnode->segidx ] = 1;
      }
    }

    // make seginfo
    struct UnusedSeg_t {
      int segidx;
      float dwall;
      std::vector<float> pos;
      bool operator<( const UnusedSeg_t& rhs ) const {
        if ( dwall<rhs.dwall ) return true;
        return false;
      };
    };

    std::vector< UnusedSeg_t > seglist_v;
    seglist_v.reserve( _segment_v.size() );

    for ( int iseg=0; iseg<(int)used_v.size(); iseg++ ) {
      if ( used_v[iseg]==1 ) continue;
      UnusedSeg_t seginfo;
      seginfo.segidx = iseg;

      int btype1 = 0;
      int btype2 = 0;
      float dwall1 = ublarcvapp::dwall_noAC( _segment_v[iseg].start, btype1 );
      float dwall2 = ublarcvapp::dwall_noAC( _segment_v[iseg].end,   btype2 );

      seginfo.dwall =  ( dwall1<dwall2 ) ? dwall1: dwall2;
      if ( dwall1<dwall2 ) {
        seginfo.pos = _segment_v[iseg].start;
      }
      else {
        seginfo.pos = _segment_v[iseg].end;
      }
      seglist_v.push_back( seginfo );
    }

    std::sort( seglist_v.begin(), seglist_v.end() );
    LARCV_DEBUG() << "number of (sorted) unused segments: " << seglist_v.size() << std::endl;

    bool something2run = true;
    int nsegtracks_made = 0;
    while (something2run) {

      // choose which segment to run
      something2run = false;
      UnusedSeg_t* seg2run = nullptr;
      
      for ( auto& seginfo : seglist_v ) {
        if ( seginfo.dwall < 30.0 && used_v[seginfo.segidx]==0 ) {
          // unused
          seg2run = &seginfo;
          something2run = true;
          break;
        }
      }

      // nothing to run
      if ( something2run ) {
        // number of tracks already made
        LARCV_DEBUG() << "Make track from segment[" << seg2run->segidx << "] dwall=" << seg2run->dwall
                      << " pos=(" << seg2run->pos[0] << "," << seg2run->pos[1] << "," << seg2run->pos[2] << ")"
                      << std::endl;
        
                         
        int ntracksmade = _track_proposal_v.size();              
        buildTracksFromPoint( seg2run->pos );
        used_v[seg2run->segidx] = 1;

        // number of tracks made
        int nmade = (int)_track_proposal_v.size()-ntracksmade;
        nsegtracks_made += nmade;
        for ( int itrack=ntracksmade; itrack<(int)_track_proposal_v.size(); itrack++ ) {
          auto& path = _track_proposal_v.at(itrack);
          for ( auto &pnode : path ) {
            used_v[pnode->segidx] = 1;
          }
        }
      }
      else {
        LARCV_DEBUG() << "No segment to run" << std::endl;
      }
      LARCV_DEBUG() << "[entry to continue]" << std::endl;
      //std::cin.get();
    }

    LARCV_DEBUG() << "number of tracks made via segment seeds: " << nsegtracks_made << std::endl;
    
  }

  /**
   * return index of segment closest to a test point
   *
   * @param[in] testpt 3D position to find closest segment to
   * @param[in] max_dist Maximum distance testpt can be from line segment and end points
   * @return Index of segment. If none within max distance, return -1
   */
  int TrackClusterBuilder::findClosestSegment( const std::vector<float>& testpt,
                                               const float max_dist )
  {
  
    // find closest segment
    float mindist = 0.;
    int   min_segidx = -1;
    
    for ( size_t iseg=0; iseg<_segment_v.size(); iseg++ ) {
      auto const& seg = _segment_v[iseg];
      float dist = pointLineDistance<float>(  seg.start, seg.end, testpt );
      float proj = pointRayProjection<float>( seg.start, seg.dir, testpt );

      if ( proj>-max_dist && proj<=seg.len+max_dist && dist<max_dist ) {
        if ( mindist>dist || min_segidx<0 ) {
          mindist = dist;
          min_segidx = iseg;
        }
      }
    }

    // if ( mindist>3.0 ) {
    //   LARCV_INFO() << "No segment is close enough to keypoint" << std::endl;
    //   return;
    // }
    
    return min_segidx;
    
  }
  
  /**
   * @brief store paths we've found as larlite::track objects in the provided event container
   *
   * We must convert a track defined as a sequence of NodePos_t instances into
   * a larlite track for storage into the output larlite ROOT file.
   *
   * This builds a track using the fitted line-segment paths.
   * This will also calculate the average dQ/dx of the larmatch hits
   * per track segment.
   *
   * We stitch by fitting at the boundaries
   *
   * @param[out] evout_track      Node paths stored in _track_proposal_v are stored into this container
   * @param[out] evout_hitcluster Hits associated to track, sorted by position along track
   * @param[in]  adc_v            Wire plane images used to get dQ/dx
   *
   */
  void TrackClusterBuilder::fillLarliteTrackContainerWithFittedTrack( larlite::event_track& evout_track,
                                                                      larlite::event_larflowcluster& evout_hitcluster,
                                                                      const std::vector<larcv::Image2D>& adc_v )
  {

    struct TrackSeg_t {
      TVector3 start;
      TVector3 end;
      TVector3 dir;
      TVector3 dqdx;
      int segidx;
    };

    for ( size_t itrack=0; itrack<_track_proposal_v.size(); itrack++ ) {

      auto const& path = _track_proposal_v[itrack];

      if ( path.size()<=1 )
        continue;

      // check that we have the track objects we need
      for ( auto const& node : path ) {
        auto const& seg = _segment_v[node->segidx];
        if ( seg.trackseg==nullptr) {
          std::stringstream ss;
          ss << "Reco Track[" << itrack << "] does not have a track object as needed" << std::endl;
          throw std::runtime_error(ss.str());
        }
      }

      // first step, collect track points, resort 
      std::vector< TrackSeg_t > seg_v;

      // also collect larflow3dhits
      larlite::larflowcluster track_hitcluster;

      std::cout << "[TrackClusterBuilder::fillLarliteTrackContainerWithFittedTrack] PATH " << itrack << std::endl;
      
      for (int inode=1; inode<path.size(); inode++ ) {

        const NodePos_t* node     = path[inode];   // end of segment
        const NodePos_t* nodeprev = path[inode-1]; // start of segment
        
        if ( node->segidx!=nodeprev->segidx ) {
          // not the node set we need
          continue;
        }

        const Segment_t& seg = _segment_v.at(node->segidx);

        const larlite::track& track = *seg.trackseg;
        const int npts = track.NumberTrajectoryPoints();
        
        int seg_reversed = 0;

        float start_dist[2] = { 0.0, 0.0 };
        for (int i=0; i<3; i++) {
          start_dist[0] += (nodeprev->pos[i]-track.LocationAtPoint(0)[i] )*( nodeprev->pos[i]-track.LocationAtPoint(0)[i] );
          start_dist[1] += (node->pos[i]-track.LocationAtPoint(0)[i] )*( node->pos[i]-track.LocationAtPoint(0)[i] );
        }
        if ( start_dist[0]>start_dist[1] ) {
          seg_reversed = 1;
        }

        int istart = 0;
        int iend = npts;
        int di = 1;
        if ( seg_reversed ) {
          istart = npts-1;
          iend = -1;
          di = -1;
        }
        for (int istep=istart; istep!=iend; istep+=di ) {
          TrackSeg_t segpt;
          segpt.start = track.LocationAtPoint(istep);
          segpt.segidx = node->segidx;
          seg_v.push_back( segpt );
        }

        for ( auto const& lfhit : *seg.cluster ) {
          track_hitcluster.push_back(lfhit);
        }
        
      }//end of loop over nodes

      std::cout << "  number of sorted track points: " << seg_v.size() << std::endl;

      // now that the points are all realigned, we can look for segment transitions
      std::vector< TrackSeg_t > stitched_v;
      int last_segidx = -1;
      int last_fill_step = -1;
      for ( int istep=0; istep<(int)seg_v.size(); istep++ ) {
        int segidx = seg_v[istep].segidx;
        if ( segidx!=last_segidx && last_segidx!=-1 ) {

          std::cout << "segment transition: " << last_segidx << "->" << segidx << std::endl;
          
          // segment transition
          int iprev_start=istep-3;
          if ( iprev_start<0 )
            iprev_start = 0;
          
          int inext=istep+2;
          if ( inext>=(int)seg_v.size() )
            inext = (int)seg_v.size()-1;

          if ( (inext-iprev_start) <=1 ) {
            std::cout << "simple transition: fill " << last_fill_step << " to " << istep << std::endl;
            // dont do anything fancy with gap
            // simply merge up to current location
            for (int jstep=last_fill_step+1; jstep<=istep; jstep++) {
              stitched_v.push_back( seg_v[jstep] );
            }
            last_fill_step = istep;
          }
          else {
            // we are going to gather the hits between the iprev_start and inext steps
            // and fit them to a new segment
            std::cout << "fit gap transition" << std::endl;
            
            // define a line segment
            std::vector< float > gap_start(3);
            std::vector< float > gap_end(3);
            std::vector< float > gap_dir(3);
            float gaplen = 0.;
            for (int i=0; i<3; i++) {
              gap_start[i] = seg_v[iprev_start].start[i];
              gap_end[i] = seg_v[inext].start[i];
              gap_dir[i] = gap_end[i]-gap_start[i];
              gaplen += gap_dir[i]*gap_dir[i];
            }
            gaplen = sqrt(gaplen);
            for (int i=0; i<3; i++)
              gap_dir[i] /= gaplen;

            // loop over cluster hits
            // of our two segments

            larflow::reco::cluster_t gapcluster;
            const larlite::larflowcluster* lfclusters[2] = { _segment_v[last_segidx].cluster, _segment_v[segidx].cluster };
            larlite::event_larflow3dhit gap_lfhit;
            for (int ic=0; ic<2; ic++) {
              for (auto const& lfhit : *(lfclusters[ic]) ) {
                std::vector<float> hit = { lfhit[0], lfhit[1], lfhit[2] };
                float s = larflow::reco::pointRayProjection<float>( gap_start, gap_dir, hit );
                //float r = larflow::reco::pointLineDistance<float>( gap_start, gap_end, hit );
                if ( s>=0 && s<=gaplen) {
                  gapcluster.points_v.push_back( hit );
                  gap_lfhit.push_back( lfhit );
                  gapcluster.hitidx_v.push_back( (int)gap_lfhit.size()-1 );
                }
              }//end of hit loop
            }//end of cluster loop
            std::cout << "gap cluster has " << gapcluster.points_v.size() << " hits" << std::endl;

            larlite::track gaptrack;
            if ( gapcluster.points_v.size()>=5 ) {
              larflow::reco::cluster_pca( gapcluster );
              gaptrack = larflow::reco::ProjectionDefectSplitter::fitLineSegmentToCluster( gapcluster, gap_lfhit, adc_v );
            }
            else {
              gaptrack.add_vertex( TVector3(gap_start[0],gap_start[1],gap_start[2]) );
              gaptrack.add_vertex( TVector3(gap_end[0],gap_end[1],gap_end[2]) );
              gaptrack.add_direction( TVector3(0,0,0) );
              gaptrack.add_direction( TVector3(0,0,0) );
            }
            int npts_gaptrack = gaptrack.NumberTrajectoryPoints();
            std::cout << "number of points in the fitted gap track: " << npts_gaptrack << std::endl;            

            // fill up to the gap
            for (int jstep=last_fill_step+1; jstep<iprev_start; jstep++) {
              stitched_v.push_back( seg_v[jstep] );
            }

            // is the fitted track in the forward or reverse direction?
            float trackdist[2] = {0,0};
            for (int i=0; i<3; i++) {
              trackdist[0] += (gaptrack.LocationAtPoint(0)[i]-gap_start[i])*(gaptrack.LocationAtPoint(0)[i]-gap_start[i]);
              trackdist[1] += (gaptrack.LocationAtPoint(npts_gaptrack-1)[i]-gap_start[i])*(gaptrack.LocationAtPoint(npts_gaptrack-1)[i]-gap_start[i]);
            }

            if ( trackdist[0]<trackdist[1] ) {          
              for (int i=0; i<npts_gaptrack; i++) {
                TrackSeg_t gappt;
                gappt.start = gaptrack.LocationAtPoint(i);
                gappt.segidx = -1; // gap label
                stitched_v.push_back( gappt );
              }
            }
            else {
              for (int i=0; i<npts_gaptrack-1; i++) {
                TrackSeg_t gappt;
                gappt.start = gaptrack.LocationAtPoint(npts_gaptrack-i-1);
                gappt.segidx = -1; // gap label
                stitched_v.push_back( gappt );
              }
            }
            std::cout << "number of hits in stitched track after gap fit: " << stitched_v.size() << std::endl;
            last_fill_step = inext;
          }//if valid gap size

        }// if at segment transition
        last_segidx = segidx;
      }//end of step loop through full track

      // last segment wont have a transition
      for (int jstep=last_fill_step+1; jstep<(int)seg_v.size(); jstep++) {
        stitched_v.push_back( seg_v[jstep] );
      }
      

      std::cout << "  number of stitched track points: " << stitched_v.size() << std::endl;      

      larlite::track lltrack;
      lltrack.reserve( stitched_v.size() );
      for (auto const& trkpt : stitched_v ) {
        lltrack.add_vertex( trkpt.start );
        lltrack.add_direction( TVector3(0,0,0) );
      }
      std::cout << "Number of track points: " << lltrack.NumberTrajectoryPoints() << std::endl;

      evout_track.emplace_back( std::move(lltrack) );
      evout_hitcluster.emplace_back( std::move(track_hitcluster) );
      
    }//end of loop over tracks
    
  }
  
}
}
