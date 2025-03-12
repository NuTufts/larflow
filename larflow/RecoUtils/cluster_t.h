#ifndef __LARFLOW_RECOUTILS_CLUSTER_T_H__
#define __LARFLOW_RECOUTILS_CLUSTER_T_H__

#include <vector>

namespace larflow {
  namespace recoutils {
    
  /**
   * @ingroup Reco
   * @struct cluster_t
   * @brief  represents cluster of space points
   */
    class cluster_t {

    public:
      cluster_t() {};
      virtual ~cluster_t() {};
    
      std::vector< std::vector<float> > points_v;        ///< vector of 3D space points in (x,y,z) coodinates
      std::vector< std::vector<int>   > imgcoord_v;      ///< vector of image coordinates (U,V,Y,tick)
      std::vector< int >                hitidx_v;        ///< vector of index of container this space point comes from
      std::vector< std::vector<float> > pca_axis_v;      ///< principle component axes
      std::vector<float>                pca_center;      ///< mean of the space points
      std::vector<float>                pca_eigenvalues; ///< eigenvalues of the principle components
      std::vector<int>                  ordered_idx_v;   ///< index of points_v, ordered by projected pos on 1st pca axis
      std::vector<float>                pca_proj_v;      ///< projection of point onto pca axis, follows ordered_idx_v
      std::vector<float>                pca_radius_v;    ///< distance of point from 1st pc axis, follows ordered_idx_v    
      std::vector< std::vector<float> > pca_ends_v;      ///< points on 1st pca-line out to the maximum projection distance from center
      std::vector< std::vector<float> > bbox_v;          ///< axis-aligned bounding box. calculated along with pca
      float                             pca_max_r;       ///< maximum radius of points from the 1st PC axis
      float                             pca_ave_r2;      ///< average r2 of points from the first PC axis
      float                             pca_len;         ///< distance between min and max points along the first PC axis
      
    };
    
  }
}

#endif
