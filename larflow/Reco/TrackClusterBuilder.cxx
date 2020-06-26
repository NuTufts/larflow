#include "TrackClusterBuilder.h"

#include "TVector3.h"
#include "larflow/Reco/geofuncs.h"

namespace larflow {
namespace reco {

  void TrackClusterBuilder::process( larcv::IOManager& iolcv,
                                     larlite::storage_manager& ioll )
  {

    std::string producer = "trackprojsplit_full";
    
    larlite::event_larflowcluster* ev_cluster
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, producer);
    larlite::event_pcaxis* ev_pcaxis
      = (larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis,producer);

    loadClusterLibrary( *ev_cluster, *ev_pcaxis );

  }

  void TrackClusterBuilder::clear()
  {
    _segment_v.clear();
    _connect_m.clear();
    _track_proposal_v.clear();
  }

  void TrackClusterBuilder::loadClusterLibrary( const larlite::event_larflowcluster& cluster_v,
                                                const larlite::event_pcaxis& pcaxis_v )
  {

    for (int i=0; i<cluster_v.size(); i++) {
      const larlite::larflowcluster& cluster = cluster_v.at(i);
      const larlite::pcaxis& pca = pcaxis_v.at(i);

      // create a segment object
      std::vector<float> start(3,0);
      std::vector<float> end(3,0);
      for (int v=0; v<3; v++){
        start[v] = pca.getEigenVectors()[3][v];
        end[v]   = pca.getEigenVectors()[4][v];
      }

      Segment_t seg( start, end );
      seg.cluster = &cluster;
      seg.pca     = &pca;
      _segment_v.push_back(seg);
      _segment_v.back().idx = (int)_segment_v.size()-1;
    }

    LARCV_INFO() << "Stored " << _segment_v.size() << " track segments" << std::endl;
  }

  /**
   * we go ahead and build all possible connections
   * this is going to be an N^2 algorithm, so use wisely.
   * we use this to develop the code. also, can use it for smallish N.
   *
   */
  void TrackClusterBuilder::buildConnections()
  {

    for (int iseg=0; iseg<(int)_segment_v.size(); iseg++ ) {
      Segment_t& seg_i = _segment_v[iseg];
      
      for (int jseg=iseg+1; jseg<(int)_segment_v.size(); jseg++) {

        // define a connection in both directions
        Segment_t& seg_j = _segment_v[jseg];
        

        // first we find the closest ends between them
        std::vector<float>* ends[2][2] = { {&seg_i.start,&seg_i.end},
                                           {&seg_j.start,&seg_j.end} };

        std::vector<float> dist(4,0);
        for (int i=0; i<2; i++) {
          for(int j=0; j<2; j++) {
            for (int v=0; v<3; v++) {
              dist[ 2*i + j ] += (ends[0][i]->at(v)-ends[1][j]->at(v))*( ends[0][i]->at(v)-ends[1][j]->at(v) );
            }
            dist[ 2*i + j ] = sqrt( dist[2*i+j] );
          }
        }

        // find the smallest dist
        int idx_small = 0;
        float min_dist = 1.0e9;
        for (int i=0; i<4; i++) {
          if ( min_dist>dist[i] ) {
            idx_small = i;
            min_dist = dist[i];
          }
        }

        std::vector<float>* end_i = ends[0][ idx_small/2 ];
        std::vector<float>* end_j = ends[1][ idx_small%2 ];

        // i->j
        Connection_t con_ij;
        con_ij.node = &seg_j;
        con_ij.from_seg_idx = iseg;
        con_ij.to_seg_idx = jseg;
        con_ij.dist = min_dist;
        con_ij.endidx = idx_small%2;
        con_ij.cosine = 0.;
        con_ij.cosine_seg = 0.;        
        con_ij.dir.resize(3,0);
        for (int v=0; v<3; v++) {
          con_ij.dir[v] = (end_j->at(v)-end_i->at(v))/min_dist;
          con_ij.cosine += (end_j->at(v)-end_i->at(v))*seg_i.dir[v]/min_dist;
          con_ij.cosine_seg += seg_i.dir[v]*seg_j.dir[v];
        }
        con_ij.cosine_seg = fabs(con_ij.cosine_seg);
        
        Connection_t con_ji;
        con_ji.node = &seg_i;
        con_ji.from_seg_idx = jseg;
        con_ji.to_seg_idx = iseg;
        con_ji.dist = min_dist;
        con_ij.endidx = idx_small/2;
        con_ji.cosine = 0.;
        con_ji.cosine_seg = con_ij.cosine_seg;
        con_ji.dir.resize(3,0);
        for (int v=0; v<3; v++) {
          con_ji.dir[v] = (end_i->at(v)-end_j->at(v))/min_dist;          
          con_ji.cosine += (end_i->at(v)-end_j->at(v))*seg_j.dir[v]/min_dist;
        }

        _connect_m[ std::pair<int,int>(iseg,jseg) ] = con_ij;
        _connect_m[ std::pair<int,int>(jseg,iseg) ] = con_ji;
        
      }      
    }

    LARCV_INFO() << "Made " << _connect_m.size() << " connections" << std::endl;
  }


  void TrackClusterBuilder::buildTracksFromPoint( const std::vector<float>& startpoint )
  {

    // find closest segment
    float mindist = 0.;
    int   min_segidx = -1;
    
    for ( size_t iseg=0; iseg<_segment_v.size(); iseg++ ) {
      auto const& seg = _segment_v[iseg];
      float dist = pointLineDistance<float>(  seg.start, seg.end, startpoint );
      float proj = pointRayProjection<float>( seg.start, seg.dir, startpoint );

      if ( proj>-3.0 && proj<=seg.len+3.0 ) {
        if ( mindist>dist || min_segidx<0 ) {
          mindist = dist;
          min_segidx = iseg;
        }
      }
    }
    if ( min_segidx<0 ) {
      LARCV_INFO() << "No acceptable segment found for startpoint" << std::endl;
      return;
    }

    if ( mindist>3.0 ) {
      LARCV_INFO() << "No segment is close enough to keypoint" << std::endl;
      return;
    }

    LARCV_DEBUG() << "Segment found idx=[" << min_segidx << "] to build from " << str(_segment_v[min_segidx]) << std::endl;

    std::vector< Segment_t* > path;
    std::vector< float > path_dir;
    std::vector< std::vector<Segment_t*> > complete_v;
    _recursiveFollowPath( _segment_v[min_segidx], path, path_dir, complete_v );

    // this will generate proposals. we must choose among them
    std::vector< std::vector<Segment_t*> > filtered_v;    
    _choose_best_paths( complete_v, filtered_v );

    for ( auto& path : filtered_v ) 
      _track_proposal_v.push_back( path );

    LARCV_INFO() << "Number of paths now stored: " << _track_proposal_v.size() << std::endl;
    
  }  

  void TrackClusterBuilder::_recursiveFollowPath( Segment_t& seg,
                                                  std::vector<Segment_t*>& path,
                                                  std::vector< float >& path_dir,
                                                  std::vector< std::vector<Segment_t*> >& complete )
  {

    // we add ourselves to the current path
    seg.inpath = true;
    path.push_back( &seg );
    // add determine the path direction (aligned or anti-aligned with the current segdir)
    if ( path.size()==1 ) {
      path_dir.push_back( 0.0 ); // undefined for first segment
    }
    else if (path.size()>1) {
      float prev_cos = 0.;
      std::vector<float> prev_center(3,0);
      std::vector<float> curr_center(3,0);
      auto& prev_seg = path[path.size()-2];
      for ( size_t v=0; v<3; v++ ) {
        prev_center[v] = 0.5*( prev_seg->start[v]+prev_seg->end[v] );
        curr_center[v] = 0.5*( seg.start[v]+seg.end[v] );
        prev_cos += ( curr_center[v]-prev_center[v] )*seg.dir[v];
      }
      if ( prev_cos>=0 )
        path_dir.push_back(1.0);
      else
        path_dir.push_back(-1.0);
    }
    //std::cout << "append new segment to stack. depth=" << path.size() << std::endl;
    //std::cin.get();
    
    // oh wow. we loop through possible connections, and descend.
    int nconnections = 0;
    float mindist = 1e9;
    float maxcos = 0;
    for (int iseg=0; iseg<_segment_v.size(); iseg++) {

      
      
      if ( iseg==seg.idx ) continue; // don't connect to yourself
      auto it = _connect_m.find( std::pair<int,int>(seg.idx,iseg) );
      if ( it==_connect_m.end() ) continue; // no connection

      //std::cout << "  (connect " << seg.idx << "->" << iseg << ") dist=" << it->second.dist << " cos="  << it->second.cosine << std::endl;
      
      // get next segment
      Segment_t& nextseg = _segment_v[iseg];
      if ( nextseg.inpath ) continue; // can't create a recursive loop

      float segcos = it->second.cosine;
      float concos = it->second.cosine_seg;
      if (path.size()<=1) segcos = fabs(segcos);
      else {
        segcos *= path_dir.back();
      }
      
      if ( mindist>it->second.dist ) mindist = it->second.dist;
      if ( maxcos<segcos ) maxcos = segcos;
      
      // criteria for accepting connection
      // ugh, the heuristics ...
      if ( (it->second.dist<2.0 && segcos>0 && concos>0.3 ) // close
           || (it->second.dist<20.0 && segcos>0.8 && concos>0.8 )  // far
           || (it->second.dist<50.0 && segcos>0.9 && concos>0.9 )
           ) {
        std::cout << "  connect segment[" << seg.idx << "] to segment[" << iseg << "] "
                  << "dist=" << it->second.dist << " "
                  << "connect-cos=" << segcos
                  << "segment-cos=" << it->second.cosine_seg
                  << std::endl;
        nconnections++;
        _recursiveFollowPath( nextseg, path, path_dir, complete );

        if ( complete.size()>=10000 ) {
          //cut this off!
          std::cout << "  cut off search. track limit reached: " << complete.size() << std::endl;
          break;
        }
      }
    }
    
    if ( nconnections==0 && complete.size()<10000 ) {
      // was a leaf, so make a complete track.
      // else was a link
      complete.push_back( path );
      LARCV_DEBUG() << "reached a leaf, copy track len=" << path.size() << " to completed list. num of completed tracks: " << complete.size() << std::endl;      
    }
    //std::cout << "  mindist=" << mindist << " maxcos=" << maxcos << std::endl;
    seg.inpath = false;
    path.pop_back();
    path_dir.pop_back();

    return;
  }

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

  void TrackClusterBuilder::_choose_best_paths( std::vector< std::vector<Segment_t*> >& complete_v,
                                                std::vector< std::vector<Segment_t*> >& filtered_v )
  {

    // we narrow down in phases
    // phase 1: we group by leaf segment, choosing min-dist path + paths within 20 cm of min dist
    // phase 2: we choose among the leaf group the highest fraction of the path length is segment lengths
    // phase 3: we choose path with smallest kick (bad to use heuristic, need more sophisticated selection)

    typedef std::vector<Segment_t*> Path_t;
    
    // phase 1, group paths by lead index
    std::map<int, std::vector< Path_t* > > leaf_groups;
    for ( auto& path : complete_v ) {
      int leafidx = path.back()->idx;
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
    float min_pathdist_ratio = 1.0e9;
    path_dist_ratio_v.reserve( leaf_groups.size() );
    for ( auto it=leaf_groups.begin(); it!=leaf_groups.end(); it++ ) {

      std::cout << "=== Find mindist subgroup in Leaf Group[" << it->first << "] ===" << std::endl;

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
        for ( size_t iseg=0; iseg<ppath->size(); iseg++ ) {

          Segment_t* pseg = ppath->at(iseg);

          pathdist += pseg->len;
          totseglen += pseg->len;

          if ( iseg+1<ppath->size() ) {
            Segment_t* pnextseg = ppath->at(iseg+1);
            auto it_con = _connect_m.find( std::pair<int,int>(pseg->idx,pnextseg->idx) );
            if ( it_con!=_connect_m.end() ) {
              pathdist += it_con->second.dist;
            }
          }
          
        }//end of loop over path segments
        std::cout << "  Leaf-group[" << it->first << "] path[" << ipath << "] pathlen=" << pathdist << " seglen=" << totseglen << std::endl;
        pathlen_v.push_back(pathdist);
        seglen_v.push_back(totseglen);
        if ( pathdist<minpathlen ) {
          minpath_idx = (int)ipath;
          minpathlen = pathdist;
        }
      }//end of path loop
      std::cout << " Leaf group minpathlen=" << minpathlen << std::endl;
      
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
      std::cout << " Leaf sub-group maxsegfrac=" << max_segfrac <<" pathidx=" << max_segfrac_idx << std::endl;
      selected_v.push_back( it->second.at(max_segfrac_idx) );
      selected_pathlen_v.push_back( pathlen_v[max_segfrac_idx] );
      
      // store path/dist ratio
      Path_t* leafpath = it->second.at(max_segfrac_idx);
      

      std::vector<float>* segends[2][2] = { { &leafpath->front()->start, &leafpath->front()->end },
                                            { &leafpath->back()->start,  &leafpath->back()->end  } };
      int front_end = 0;
      int back_end = 0;
      float maxdist = 0;
      for (int i=0; i<2; i++) {
        for ( int j=0; j<2; j++ ) {
          float end_dist = 0.;
          for (int v=0; v<3; v++)
            end_dist += ( segends[0][i]->at(v) - segends[1][j]->at(v) )*( segends[0][i]->at(v) - segends[1][j]->at(v) );
          end_dist = sqrt(end_dist);
          if ( maxdist<end_dist ) {
            maxdist = end_dist;
            front_end = i;
            back_end  = j;
          }
        }
      }

      float pathdist_ratio = 0.;
      if ( maxdist>0 ) {
        pathdist_ratio = pathlen_v[max_segfrac_idx]/maxdist;
      }
      path_dist_ratio_v.push_back( pathdist_ratio );

      if ( pathdist_ratio>0 && pathdist_ratio<min_pathdist_ratio ) {
        min_pathdist_ratio = pathdist_ratio;
      }
      
      std::cout << "  Leafgroup["  << it->first << "] best path dist between ends: " << maxdist << std::endl;
      
      std::cout << "=== End of Leafgroup[" << it->first << "] selection path/dist ratio: " << pathdist_ratio << " len=" << pathlen_v[max_segfrac_idx] << " ===" << std::endl;
    }//end of group loop


    // choose among the best of the best?
    // the minimum ratio of path-length to 
    float min_max_len = 0.;
    int min_max_len_idx = 0;
    for ( size_t i=0; i<selected_v.size(); i++ ) {
      if ( path_dist_ratio_v[i]>=1.0 && path_dist_ratio_v[i]<min_pathdist_ratio+0.2 ) {
        if ( min_max_len<selected_pathlen_v[i] ) {
          min_max_len_idx = i;
          min_max_len = selected_pathlen_v[i];
        }
      }
    }

    filtered_v.push_back( *selected_v[min_max_len_idx] );
    std::cout << "Number of leaf-group candidate tracks return: " << (int)filtered_v.size()-nfiltered_before << std::endl;    
  }

  void TrackClusterBuilder::fillLarliteTrackContainer( larlite::event_track& evout_track )
  {

    for ( size_t itrack=0; itrack<_track_proposal_v.size(); itrack++ ) {

      auto const& path = _track_proposal_v[itrack];
      
      larlite::track lltrack;
      lltrack.reserve(2*path.size()); // npts = 2*num seg;

      std::vector<float> last_seg_dir(3,0);
      for (size_t iseg=0; iseg<path.size(); iseg++ ) {

        const Segment_t* seg = path[iseg];

        if ( iseg+1<path.size() ) {
          const Segment_t* nextseg = path[iseg+1];
          auto it = _connect_m.find( std::pair<int,int>( seg->idx, nextseg->idx ) );
          Connection_t& con = it->second;
          std::vector<float> center(3,0);
          for (int v=0; v<3; v++) center[v] = 0.5*(nextseg->start[v]+nextseg->end[v]);
          float proj = pointRayProjection<float>( seg->start, seg->dir, center );
          if ( proj<0 ) {
            // go end to start
            lltrack.add_vertex( TVector3(seg->end[0],seg->end[1],seg->end[2]) );
            lltrack.add_vertex( TVector3(seg->start[0],seg->start[1],seg->start[2]) );
            lltrack.add_direction( TVector3(-seg->dir[0],-seg->dir[1],-seg->dir[2]) );
            lltrack.add_direction( TVector3(con.dir[0], con.dir[1], con.dir[2] ) );
          }
          else {
            // go start to end
            lltrack.add_vertex( TVector3(seg->start[0],seg->start[1],seg->start[2]) );
            lltrack.add_vertex( TVector3(seg->end[0],seg->end[1],seg->end[2]) );            
            lltrack.add_direction( TVector3(seg->dir[0],seg->dir[1],seg->dir[2]) );
            lltrack.add_direction( TVector3(con.dir[0], con.dir[1], con.dir[2] ) );            
          }
          last_seg_dir = con.dir;
        }//all but last segment
        else {
          float segcos = 0.;
          for (int i=0; i<3; i++ )
            segcos += last_seg_dir[i]*seg->dir[i];

          if ( segcos<0 ) {
            // go end to start
            lltrack.add_vertex( TVector3(seg->end[0],seg->end[1],seg->end[2]) );
            lltrack.add_vertex( TVector3(seg->start[0],seg->start[1],seg->start[2]) );
            lltrack.add_direction( TVector3(-seg->dir[0],-seg->dir[1],-seg->dir[2]) );
            lltrack.add_direction( TVector3(-seg->dir[0],-seg->dir[1],-seg->dir[2]) );
          }
          else {
            // go start to end
            lltrack.add_vertex( TVector3(seg->start[0],seg->start[1],seg->start[2]) );
            lltrack.add_vertex( TVector3(seg->end[0],seg->end[1],seg->end[2]) );            
            lltrack.add_direction( TVector3(seg->dir[0],seg->dir[1],seg->dir[2]) );
            lltrack.add_direction( TVector3(seg->dir[0],seg->dir[1],seg->dir[2]) );            
          }
        }
        
      }//end of loop over segments

      evout_track.emplace_back( std::move(lltrack) );
    }//end of loop over tracks
    
  }
  
}
}
