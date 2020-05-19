#include "KPTrackFit.h"


#include <cilantro/kd_tree.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_utility.hpp>

#include <set>
#include <algorithm>


namespace larflow {
namespace reco {

  std::vector<int> KPTrackFit::fit( const std::vector< std::vector<float> >& point_v,
                                    int start, int end )
  {

    // use kdtree to make edges between points
    std::map< std::pair<int,int>, float > distmap;
    defineEdges( point_v, distmap, 10.0 );

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
    int curr = point_v.size()-1;
    int pred = G[curr].pred;


    std::vector<int> path_idx;
    path_idx.push_back( G[curr].index );    

    float totdist = 0.;
    while ( pred!=0 && curr!=pred ) {
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
      totdist += realdist;
      path_idx.push_back( G[pred].index );
      curr = pred;
      pred = G[curr].pred;
    }
    LARCV_DEBUG() << "end of path. npoints=" << path_idx.size() << " totlength=" << totdist << std::endl;

    return path_idx;
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

}
}
