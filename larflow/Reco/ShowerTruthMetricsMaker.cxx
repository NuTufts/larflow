#include "ShowerTruthMetricsMaker.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  /**
   * @brief Returns MCPixelPGraph's reanalysis of true edep position using larlite and larcv truth information
   *
   */ 
  std::vector<float>
  ShowerTruthMetricsMaker::getPhotonFirstEDepPosition( ublarcvapp::mctools::MCPixelPGraph& mcpg,
						       const int trackid_of_shower )
  {
    std::vector<float> pos(3,0);

    // first get the node with the given track id
    auto pnode = mcpg.findTrackID( trackid_of_shower );

    if ( pnode==nullptr ) {
      std::stringstream msg;
      msg << "[larflow/Reco/ShowerTruthMetricsMaker.cxx.L21][ getPhotonFirstEDepPosition() ] "
	  << " Could not find a particle with trackid=" << trackid_of_shower << std::endl;
      throw std::runtime_error(msg.str());
    }

    if ( pnode->pid!=22 ) {
      std::stringstream msg;
      msg << "[larflow/Reco/ShowerTruthMetricsMaker.cxx.L28][ getPhotonFirstEDepPosition() ] "
	  << " Particle with trackid=" << trackid_of_shower << " is not a photon. "
	  << " PDGcode=" << pnode->pid << std::endl;
    }

    for (int v=0; v<3; v++) {
      pos[v] = pnode->first_edep_pos[v];
    }
    
    return pos;
  }

  // std::vector<float>
  // ShowerTruthMetricsMaker::getPhotonFirstEDepPosition( const ublarcvapp::mctools::MCPixelPGraph& mcpg,
  // 						       const int trackid_of_shower )
  // {
  //   std::vector<float> pos(3,0);
  //   return pos;
  // }

  std::vector<float>
  ShowerTruthMetricsMaker::getPhotonTrunkLineSegment( ublarcvapp::mctools::MCPixelPGraph& mcpg,
						      const int trackid_of_shower )
  {

    // first get the node with the given track id
    auto pnode = mcpg.findTrackID( trackid_of_shower );

    if ( pnode==nullptr ) {
      std::stringstream msg;
      msg << "[larflow/Reco/ShowerTruthMetricsMaker.cxx.L21][ getPhotonFirstEDepPosition() ] "
	  << " Could not find a particle with trackid=" << trackid_of_shower << std::endl;
      throw std::runtime_error(msg.str());
    }

    if ( pnode->pid!=22 ) {
      std::stringstream msg;
      msg << "[larflow/Reco/ShowerTruthMetricsMaker.cxx.L28][ getPhotonFirstEDepPosition() ] "
	  << " Particle with trackid=" << trackid_of_shower << " is not a photon. "
	  << " PDGcode=" << pnode->pid << std::endl;
    }

    std::vector<float> trunk_segment(6,0);
    const ublarcvapp::mctools::MCPixelPGraph::pointList& trunk_point_v = mcpg.getTruePhotonTrunk3DPoints( (*pnode) );
    

    // pass points into larflow::reco::cluster
    larflow::reco::cluster_t cluster;
    cluster.points_v.reserve( trunk_point_v.size() );
    cluster.hitidx_v.reserve( trunk_point_v.size() );
    for ( int ipt=0; ipt<(int)trunk_point_v.size(); ipt++ ) {
      cluster.points_v.push_back( trunk_point_v[ipt] );
      cluster.hitidx_v.push_back( ipt );
    }
    // we run the pca algorithm made for our larflow::reco::cluster objects
    larflow::reco::cluster_pca( cluster );

    // decide which of the first pca-axis projection points is closer to start of cluster
    float dist[2] = {0,0};
    for (int i=0; i<2; i++) {
      for (int v=0; v<3; v++) {
	dist[i] += (cluster.pca_ends_v[i][v]-pnode->first_edep_pos[v])*(cluster.pca_ends_v[i][v]-pnode->first_edep_pos[v]);
      }
    }
    int istart = 0;
    int iend   = 1;
    if ( dist[0]>dist[1] ) {
      istart = 1;
      iend = 0;
    }
    for (int v=0; v<3; v++) {
      trunk_segment[v]   = cluster.pca_ends_v[istart][v];
      trunk_segment[3+v] = cluster.pca_ends_v[iend][v];
    }
    
    
    return trunk_segment;
  }
  
  std::vector<float>
  ShowerTruthMetricsMaker::getPhotonTrunkPlanePixelSum( ublarcvapp::mctools::MCPixelPGraph& mcpg,
							const int trackid_of_shower )
  {
    std::vector<float> pixelsum(3,0);
    pixelsum = mcpg.getTruePhotonTrunkPlanePixelSums( trackid_of_shower );
    return pixelsum;
  }
  
}
}
