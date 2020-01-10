#ifndef __LARFLOW_RECO_CLUSTER_FUNCTIONS_H__
#define __LARFLOW_RECO_CLUSTER_FUNCTIONS_H__

#include <vector>
#include <string>
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"

#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/ImageMeta.h"

namespace larflow {
namespace reco {

  /**
   * cluster used in the functions below
   *
   */
  struct cluster_t {
    
    std::vector< std::vector<float> > points_v;
    std::vector< std::vector<int>   > imgcoord_v;
    std::vector< int >                hitidx_v;      // index of larflow hit this point comes from
    std::vector< std::vector<float> > pca_axis_v;
    std::vector<float>                pca_center;
    std::vector<float>                pca_eigenvalues;
    std::vector<int>                  ordered_idx_v; // index of points, ordered by projected pos on 1st pca axis
    std::vector<float>                pca_proj_v;    // projection of point onto pca axis, follows ordered_idx_v
    std::vector<float>                pca_radius_v;  // distance of point from 1st pc axis, follows ordered_idx_v    
    std::vector< std::vector<float> > pca_ends_v;    // points on 1st pca-line out to the maximum projection distance from center
    float                             pca_max_r;
    float                             pca_ave_r2;
    float                             pca_len;

  };
  
  void cluster_larflow3dhits( const std::vector<larlite::larflow3dhit>& hit_v,
                              std::vector< cluster_t >& cluster_v,
                              const float maxdist=5.0, const int minsize=5, const int maxkd=5 );

  void cluster_dump2jsonfile( const std::vector<cluster_t>& cluster_v,
                              std::string outfilename );  

  void cluster_pca( cluster_t& cluster );

  void cluster_runpca( std::vector<cluster_t>& cluster_v );

  void cluster_splitbytrackshower( const std::vector<larlite::larflow3dhit>& hit_v,
                                   const std::vector<larcv::Image2D>& ssnettrack_image_v,
                                   std::vector<larlite::larflow3dhit>& track_hit_v,
                                   std::vector<larlite::larflow3dhit>& shower_hit_v );

  void cluster_imageprojection( const cluster_t& cluster, std::vector<larcv::Image2D>& clust2d_images_v );

  void cluster_getcontours( std::vector<larcv::Image2D>& clust2d_images_v );

  float cluster_closest_endpt_dist( const cluster_t& clusta, const cluster_t& clust_b,
                                    std::vector< std::vector<float> >& endpts );
  
  float cluster_cospca( const cluster_t& clusta, const cluster_t& clustb );

  cluster_t cluster_merge( const cluster_t& clust_a, const cluster_t& clust_b );

  larlite::pcaxis cluster_make_pcaxis( const cluster_t& cluster, int id=0 );

  cluster_t cluster_from_larflowcluster( const larlite::larflowcluster& lfcluster );
  
}
}

#endif
