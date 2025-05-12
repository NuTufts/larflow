#include "KPCluster.h"
#include <iostream>

#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/Geometry.h"

namespace larflow {
namespace reco {

  /**
   * @brief Print cluster info to standard out
   */  
  void KPCluster::printInfo() const
  {
    std::cout << "[KPCluster] type=" << _cluster_type << std::endl;
    std::cout << " center: (" << center_pt_v[0] << "," << center_pt_v[1] << "," << center_pt_v[2] << ")" << std::endl;
    std::cout << " num points: " << pt_pos_v.size() << std::endl;
    std::cout << " max score: " << max_score << std::endl;
    for (int i=0; i<3; i++ ) {
      std::cout << " pca-" << i << ": val=" << pca_eigenvalues[i]
                << " dir=(" << pca_axis_v[i][0] << "," << pca_axis_v[i][1] << "," << pca_axis_v[i][2]  <<")" << std::endl;
    }
  }

  /**
   * @brief get information in KPCluster in larlite::larflow3dhit form
   *
   */
  larlite::larflow3dhit KPCluster::as_larflow_hit() const
  {

    larlite::larflow3dhit hit;
    std::vector<double> vtxpos(3);
    hit.resize( 5, 0 ); // [0-2]: hit pos, [3]: type, [4]: max net score
    for (int i=0; i<3; i++) {
      hit[i] = max_pt_v[i]; // use hit with maximum keypoint score
      //hit[i] = center_avg_pt_v[i]; // use (keypoint score)^2 weighted position.
      //hit[i] = center_pt_v[i]; // use Gaussian fit position (not good, deprecated)
      vtxpos[i] = max_pt_v[i];
    }
    hit[3] = _cluster_type;
    hit[4] = max_score;
    
    hit.targetwire.resize( 3, 0 );
    for  (int p=0; p<3; p++) 
      hit.targetwire[p] = larutil::Geometry::GetME()->WireCoordinate( vtxpos, p );
    hit.tick = vtxpos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;

    return hit;    
  }

  larlite::pcaxis KPCluster::get_pcaxis( int cidx ) const
  {

    // pca-axis
    larlite::pcaxis::EigenVectors e_v;
    // just std::vector< std::vector<double> >
    // we store axes (3) and then the 1st axis end points. So five vectors.
    for ( auto const& a_v : pca_axis_v ) {
      std::vector<double> da_v = { (double)a_v[0], (double)a_v[1], (double) a_v[2] };
      e_v.push_back( da_v );
    }
    // start and end points
    for ( auto const& p_v : pca_ends_v ) {
      std::vector<double> dp_v = { (double)p_v[0], (double)p_v[1], (double)p_v[2] };
      e_v.push_back( dp_v );
    }
    double eigenval[3] = { pca_eigenvalues[0], pca_eigenvalues[1], pca_eigenvalues[2] };
    double centroid[3] = { pca_center[0], pca_center[1], pca_center[2] };
    larlite::pcaxis llpca( true, pt_pos_v.size(), eigenval, e_v, centroid, 0, cidx);

    return llpca;
  }

}
}
