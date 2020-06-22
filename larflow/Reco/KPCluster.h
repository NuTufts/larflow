#ifndef __KPCLUSTER_H__
#define __KPCLUSTER_H__

#include <vector>

namespace larflow {
namespace reco {

  class KPCluster {

  public:

    KPCluster()
      : center_pt_v({0,0,0}),
      pca_max_r(-1.0),
      pca_ave_r2(-1.0),
      pca_len(-1.0),
      max_idx(-1),
      max_score(-1.0),
      max_pt_v( {0,0,0} ),
      _cluster_idx(-1),
      _cluster_type(-1)
      {};
    virtual ~KPCluster() {};
    
    std::vector< float > center_pt_v;             ///< center point
    std::vector< std::vector<float> > pt_pos_v;   ///< points associated to the center
    std::vector< float >              pt_score_v; ///< score of cluster points

    // pca info from clustering of neighboring points
    // this info is copied from the cluster_t object used to make it
    std::vector< std::vector<float> > pca_axis_v;
    std::vector<float>                pca_center;
    std::vector<float>                pca_eigenvalues;
    std::vector< std::vector<float> > pca_ends_v;    // points on 1st pca-line out to the maximum projection distance from center
    std::vector< std::vector<float> > bbox_v;        // axis-aligned bounding box. calculated along with pca
    float                             pca_max_r;
    float                             pca_ave_r2;
    float                             pca_len;

    int                               max_idx;       //< hit in cluster with maximum score
    float                             max_score;     //< maximum score of hit in cluster
    std::vector<float>                max_pt_v;      //< position of maximum point

    int _cluster_idx;                             ///< associated cluster_t in KeypointReco::_cluster_v (internal use only)
    //cluster_t cluster;

    int _cluster_type;
    
    void printInfo() const;

  };
  
}
}

#endif
