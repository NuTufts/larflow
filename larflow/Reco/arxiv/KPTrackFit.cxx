#include "KPTrackFit.h"

#include <cilantro/kd_tree.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_utility.hpp>

#include <set>
#include <algorithm>

// larlite
#include "LArUtil/Geometry.h"

#include "TRandom3.h"
#include "geofuncs.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  std::vector<int> KPTrackFit::fit( const std::vector< std::vector<float> >& all_point_v,
                                    const std::vector< larcv::Image2D >& badch_v,
                                    int start, int end )
  {


    TRandom3 rand(314159);

    // we thing out the points. aiming for 3 pts per cm on average.
    float linedist = 0.;
    for (int i=0; i<3; i++) {
      linedist += (all_point_v[start][i]-all_point_v[end][i])*(all_point_v[start][i]-all_point_v[end][i]);
    }
    linedist = sqrt(linedist);

    LARCV_NORMAL() << "Fitting " << all_point_v.size() << " given points." << std::endl;
    
    std::vector< std::vector<float> > point_v;
    std::vector< int >                point_idx_v;
    int npts = (int)linedist*5.0;
    float frac = (float)npts/(float)all_point_v.size();
    for (int i=0; i<(int)all_point_v.size(); i++) {
      if ( frac>=1.0 || rand.Uniform()<frac || i==start || i==end ) {
        point_v.push_back( all_point_v[i] );
        point_idx_v.push_back( i );
      }
    }

    LARCV_NORMAL() << "Fitting " << point_v.size() << " sampled points." << std::endl;    

    // use kdtree to make edges between points
    std::map< std::pair<int,int>, float > distmap;
    defineEdges( point_v, distmap, 20.0 );

    addConnectGapsViaClusterPCA( point_v, 5.0, 50.0, distmap );
    
    //addBadChCrossingConnections( point_v, badch_v, 10.0, 30.0, distmap );

    // Define Vertex and Edge objects
    struct VertexData {
      int index;
      int pred;
      int dist;
      std::vector<float> pos;
    };

    struct EdgeData {
      float dist;
      float realdist;
    };

    // define boost graph
    typedef boost::adjacency_list<boost::vecS, boost::vecS,
                                  boost::directedS,
                                  VertexData, EdgeData> GraphDefinition_t;

    // make graph and define vertex values
    GraphDefinition_t G(point_v.size());
    for ( size_t inode=0; inode<point_v.size(); inode++ ) {
      auto const& pt = point_v[inode];
      G[inode].index = inode;
      G[inode].pos = { pt[0], pt[1], pt[2] };
    }

    // define edges
    for ( auto it=distmap.begin(); it!=distmap.end(); it++ ) {
      const std::pair<int,int>&  key = it->first;
      const float realdist  = it->second;
      //auto edge = add_edge( key.first, key.second, { dist, realdist }, G ).first;
      auto edge = add_edge( key.first, key.second, { realdist, realdist }, G ).first;
    }

    if (logger().debug()) {
      LARCV_DEBUG() << "Dump graph" << std::endl;      
      boost::print_graph(G, boost::get(&VertexData::index, G));
    }
    
    dijkstra_shortest_paths(G, 0,
                            predecessor_map(get(&VertexData::pred, G))
                            .distance_map(get(&VertexData::dist, G))
                            .weight_map(get(&EdgeData::dist, G)));

    

    // we can trace out the path back to the root node.
    // we find the longest gap and the total real distance.
    LARCV_DEBUG() << "node dump after dijkstra." << std::endl;
    for ( size_t i=0; i<point_v.size(); i++ ) {
      LARCV_DEBUG() << "   node[" << i << "] index=" << G[i].index
                    << " pos=(" << G[i].pos[0] << "," << G[i].pos[1] << "," << G[i].pos[2] << ") "
                    << " pred=" << G[i].pred << " dist=" << G[i].dist << std::endl;
    }
    
    LARCV_DEBUG() << "Ran dijkstra_shortest_path. Trace out path from end node to start node." << std::endl;    
    int curr = (int)point_v.size()-1;
    std::vector<int> path_idx;
    int pred = G[curr].pred;
    
    float totdist = 0.;
    while ( curr!=0 && pred!=curr ) {
      //bool exists = boost::edge( curr, pred, G ).second;      
      //auto edge = boost::edge( curr, pred, G ).first;
      // if ( !exists ) {
      //   LARCV_CRITICAL() << "edge between node index [" << curr << "] -> [" << pred << "] does not exist" << std::endl;
      //   std::stringstream msg;
      //   msg << __FILE__ << ":" << __LINE__ << std::endl;
      //   throw std::runtime_error(msg.str());        
      //   break;
      // }

      std::pair<int,int> key(curr,pred);
      auto it_real = distmap.find(key);
      float realdist = -1;
      if ( it_real!=distmap.end() )   realdist = it_real->second;
      LARCV_DEBUG() << "    node[" << curr <<"] -> pred[" << pred << "] "
                    << " pos=(" << G[curr].pos[0] << "," << G[curr].pos[1] << "," << G[curr].pos[2] << ") "
                    << " dist=" << realdist
                    << std::endl;
      //<< " gapdist="  << get(&EdgeData::dist, G)[edge]
      //          << " realdist=" << get(&EdgeData::realdist,G)[edge] << std::endl;
      if ( realdist>0 ) {
        totdist += realdist;
        path_idx.push_back( G[curr].index );
      }
      curr = G[curr].pred;
      pred = G[curr].pred;
    }
    LARCV_DEBUG() << "end of path. npoints=" << path_idx.size() << " totlength=" << totdist << std::endl;
    path_idx.push_back( 0 );

    if ( frac>=1.0 )
      return path_idx;

    // else we need to go back to the unsampled index
    std::vector<int> unsampled_idx;
    unsampled_idx.reserve( path_idx.size() );
    for ( auto& subidx : path_idx ) {
      unsampled_idx.push_back( point_idx_v[ subidx ] );
    }

    return unsampled_idx;
  }
  
  
  void KPTrackFit::defineEdges( const std::vector< std::vector<float> >& point_v,
                                std::map< std::pair<int,int>, float >& distmap,       // distance between nodes
                                const float max_neighbor_dist )   // pixel gap between vertices
  {
    
    // convert into eigen    
    std::vector< Eigen::Vector3f > epoint_v;
    for ( auto const& pt : point_v ) {
      Eigen::Vector3f ept( pt[0], pt[1], pt[2] );
      epoint_v.emplace_back( std::move(ept) );
    }

    distmap.clear();

    // use a kd tree to define sparse adjacency matrix
    cilantro::KDTree3f tree(epoint_v);

    size_t max_n_neighbors = 20;
    float nedges_per_vertex = 0;
    
    for ( size_t idx=0; idx<epoint_v.size(); idx++ ) {
      auto& vertex = epoint_v.at(idx);
      cilantro::NeighborSet<float> nn;
      tree.kNNInRadiusSearch(vertex, max_n_neighbors, max_neighbor_dist, nn );
      //tree.radiusSearch( vertex, max_neighbor_dist*max_neighbor_dist, nn );

      int numneighbors = 0;
      for ( size_t inn=0; inn<nn.size(); inn++ ) {
        auto& neighbor = nn.at(inn);
        if ( idx==neighbor.index )
          continue; // no self-connections

        if ( distmap.find( std::pair<int,int>(idx,neighbor.index) )!=distmap.end() ) {
          // already defined
          continue;
        }
        
        auto& neighbornode = point_v[ neighbor.index ];
        numneighbors++;

        distmap[ std::pair<int,int>( (int)idx, (int)neighbor.index) ]    = sqrt(neighbor.value);
        //distmap[ std::pair<int,int>( (int)neighbor.index, (int)idx) ]    = sqrt(neighbor.value);
      }
      nedges_per_vertex += (float)numneighbors;
    }
    nedges_per_vertex /= (float)point_v.size();
    LARCV_NORMAL() << "[KPTrackFit::defineEdges] Defined graph edges (size=" << distmap.size() << "). ave neighors per node=" << nedges_per_vertex << std::endl;
    
  }

  void KPTrackFit::addBadChCrossingConnections( const std::vector< std::vector<float> >& point_v,
                                                const std::vector< larcv::Image2D >& badch_v,
                                                const float min_gap, const float max_gap,
                                                std::map< std::pair<int,int>, float >& distmap )
  {

    struct NearDeadPix_t {
      int idx;
      float r;
      bool operator< (const NearDeadPix_t& rhs ) {
        if ( r<rhs.r ) return true;
        return false;
      };
    };
    
    std::vector<NearDeadPix_t> neardead_idx_v;
    
    for ( int idx=0; idx<(int)point_v.size(); idx++ ) {
      // check if point is near dead region
      auto const& pt = point_v[idx];
      int nplane_neardead = 0;
      for ( int p=0; p<3; p++ ) {
        int wire = (int)larutil::Geometry::GetME()->NearestWire( std::vector<double>( {pt[0],pt[1],pt[2]} ), p );
        for (int dc=0; dc<=0; dc++) {
          int c = wire+dc;
          if ( c<0 || c>=(int)badch_v[p].meta().cols() ) continue;
          if ( badch_v[p].pixel(0,c)>1 ) {
            nplane_neardead++;
            break;
          }
        }
      }

      float r = pointLineDistance( point_v.front(), point_v.back(), pt );
      
      if ( nplane_neardead>0 && r<20.0 ) {
        NearDeadPix_t pix;
        pix.idx = idx;
        pix.r = r;
        neardead_idx_v.push_back( pix );
      }
    }

    sort( neardead_idx_v.begin(), neardead_idx_v.end() );

    LARCV_INFO() << "points near dead regions: " << neardead_idx_v.size() << std::endl;

    // we have to limit the pixels we consider
    int ndead = (int)neardead_idx_v.size();
    if ( ndead>1000 )
      ndead = 1000;
    
    const float max_step_size = 1.0;
    
    // N^2 comparison. boo. only connect points with gap larger than `min_gap` parameter
    int nadded = 0;
    int nalready_connected = 0;
    int npairs = -1;
    for (int i=0; i<ndead; i++ ) {
      for (int j=i+1; j<ndead; j++) {

        npairs++;

        if ( npairs%10000==0 ) LARCV_DEBUG() << "neardead pair " << npairs << std::endl;

        int idx_i = neardead_idx_v[i].idx;
        int idx_j = neardead_idx_v[j].idx;
        
        std::vector<float> pt_i = point_v[ idx_i ];
        std::vector<float> pt_j = point_v[ idx_j ];

        std::pair<int,int> pairij( idx_i, idx_j );
        std::pair<int,int> pairji( idx_j, idx_i );
        
        if ( distmap.find( pairij )!=distmap.end() || distmap.find( pairji )!=distmap.end() ){
          nalready_connected++;
          continue;
        }

        float dist = 0.;
        std::vector<float> gapdir(3,0);
        for (int v=0; v<3; v++) {
          gapdir[v] = pt_j[v]-pt_i[v];
          dist += gapdir[v]*gapdir[v];
        }
        dist = sqrt(dist);

        if ( dist<min_gap || dist>max_gap) continue;

        for (int v=0; v<3; v++ ) gapdir[v] /= dist;

        bool connect = false;
        int nsteps_in_dead = 0;
        int nsteps = dist/max_step_size+1;
        float stepsize = dist/float(nsteps);

        for (int istep=1; istep<=nsteps; istep++) {
          std::vector<double> pt(3,0);
          for (int v=0; v<3; v++) pt[v] = (double)pt_i[v] + (double)stepsize*gapdir[v];
          bool indead = false;
          for (int p=0; p<3; p++) {
            int wire = (int)larutil::Geometry::GetME()->NearestWire( pt, p );
            if ( wire<0 || wire>=(int)badch_v[p].meta().cols() ) continue;
            if ( badch_v[p].pixel(0,wire)>0 ){
              indead = true;
            }
            if ( indead ) break;
          }
          if ( indead ) nsteps_in_dead++;
          if ( nsteps_in_dead>=nsteps/2 ) {
            connect = true;
            break;
          }
        }


        if ( connect ) {
          //LARCV_INFO() << "Added dead channel gap connection, dist=" << dist << std::endl;
          distmap[ pairij ] = dist;
          distmap[ pairji ] = dist;
          nadded++;
        }
        
      }
    }
    LARCV_INFO() << "Added " << nadded << " dead channel connections. "
                 << "Already added: " << nalready_connected
                 << std::endl;

    
  }

  /**
   * 
   * @param[in] point_v  Points we are considering
   */
  void KPTrackFit::addConnectGapsViaClusterPCA( const std::vector< std::vector<float> >& point_v,
                                                const float min_gap, const float max_gap,
                                                std::map< std::pair<int,int>, float >& distmap )
  {

    std::vector<cluster_t> cluster_v;
    cluster_spacepoint_v( point_v, cluster_v );

    // get pca for clusters
    for ( auto& c : cluster_v ) {
      cluster_pca(c);
    }

    int nadded = 0;
    
    // we do a N^2 comparison, checking for gaps
    for ( size_t i=0; i<cluster_v.size(); i++ ) {
      auto& c_i = cluster_v[i];

      // too small to bother
      if (c_i.points_v.size()<10) continue;
      
      for ( size_t j=i+1; j<cluster_v.size(); j++) {
        auto& c_j = cluster_v[j];
        // too small to bother
        if (c_i.points_v.size()<10) continue;

        // check end point distance
        std::vector< std::vector<float> > endpts;
        float endptdist = cluster_closest_endpt_dist( c_i, c_j, endpts );

        if ( endptdist>max_gap || endptdist<min_gap )
          continue;
        
        float cospca = cluster_cospca( c_i, c_j );

        if ( fabs(cospca) < 0.5 )
          continue;

        // ok looks like something we should connect
        // we connect points near the endpts
        // we use the project on the axis to define near the end

        // cluster i
        // ----------
        std::vector<float> endpt_s_i;
        for ( auto& endpt : endpts ) {
          float s=0;
          for (int v=0; v<3; v++ ) s += (endpt[v]-c_i.pca_center[v])*c_i.pca_axis_v[0][v];
          endpt_s_i.push_back(s);
        }

        std::set<int> end_idx_i_v;
        for ( size_t iidx=0; iidx<c_i.points_v.size(); iidx++ ) {
          for ( auto& end_s : endpt_s_i ) {
            float ds = fabs( end_s - c_i.pca_proj_v[ c_i.ordered_idx_v[iidx] ] );
            if (ds<15.0)
              end_idx_i_v.insert( c_i.hitidx_v[iidx] );
          }
        }

        // cluster j
        // ----------
        std::vector<float> endpt_s_j;
        for ( auto& endpt : endpts ) {
          float s=0;
          for (int v=0; v<3; v++ ) s += (endpt[v]-c_j.pca_center[v])*c_j.pca_axis_v[0][v];
          endpt_s_j.push_back(s);
        }

        std::set<int> end_idx_j_v;
        for ( size_t jidx=0; jidx<c_j.points_v.size(); jidx++ ) {
          for ( auto& end_s : endpt_s_j ) {
            float ds = fabs( end_s - c_j.pca_proj_v[ c_j.ordered_idx_v[jidx] ] );
            if (ds<15.0)
              end_idx_j_v.insert( c_j.hitidx_v[jidx] );
          }
        }
        
        for ( auto& iidx : end_idx_i_v ) {
          for ( auto& jidx : end_idx_j_v ) {

            std::pair<int,int> apair(iidx,jidx);
            std::pair<int,int> bpair(jidx,iidx);

            if ( distmap.find(apair)==distmap.end()
                 || distmap.find(bpair)==distmap.end() ) {

              float dist=0.;
              for (int v=0; v<3; v++) {
                dist += ( point_v[iidx][v]-point_v[jidx][v] )*( point_v[iidx][v]-point_v[jidx][v] );
              }
              dist = sqrt(dist);

              distmap[apair] = dist;
              distmap[bpair] = dist;
              nadded++;
            }
            
          }
        }
        
        
      }//end of loop over j cluster
      
    }//end of loop over i cluster

    LARCV_INFO() << "number of gap connections created: " << nadded << std::endl;
  }
}
}
