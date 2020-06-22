#include "KPCluster.h"
#include <iostream>

namespace larflow {
namespace reco {

  /**
   * Print cluster info to standard out
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

}
}
