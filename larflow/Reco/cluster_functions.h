#ifndef __LARFLOW_RECO_CLUSTER_FUNCTIONS_H__
#define __LARFLOW_RECO_CLUSTER_FUNCTIONS_H__

#include <vector>
#include <string>
#include "DataFormat/larflow3dhit.h"

namespace larflow {
namespace reco {

  /**
   * cluster used in the functions below
   *
   */
  struct cluster_t {
    
    std::vector< std::vector<float> > points_v;
    std::vector< std::vector<float> > pca_axis_v;
    std::vector<float>                pca_center;
    std::vector<float>                pca_eigenvalues;
    std::vector<int>                  ordered_idx_v; // index of points, ordered by projected pos on 1st pca axis
    std::vector<float>                pca_proj_v;    // projection of point onto pca axis, follows ordered_idx_v
    std::vector< std::vector<float> > pca_ends_v;    // points on 1st pca-line out to the maximum projection distance from center

  };
  
  void cluster_larflow3dhits( const std::vector<larlite::larflow3dhit>& hit_v,
                              std::vector< cluster_t >& cluster_v );

  void cluster_dump2jsonfile( const std::vector<cluster_t>& cluster_v,
                              std::string outfilename );  

  void cluster_pca( cluster_t& cluster );

  void cluster_runpca( std::vector<cluster_t>& cluster_v );
    
}
}

#endif
