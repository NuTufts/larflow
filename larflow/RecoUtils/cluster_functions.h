#ifndef __LARFLOW_RECO_CLUSTER_FUNCTIONS_H__
#define __LARFLOW_RECO_CLUSTER_FUNCTIONS_H__

#include <vector>
#include <string>
#include "larlite/DataFormat/larflow3dhit.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/pcaxis.h"
#include "larlite/DataFormat/track.h"

#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/ImageMeta.h"

#ifndef __CINT__
#ifndef __CLING__
#include "nlohmann/json.hpp"
#endif
#endif

#include "larflow/RecoUtils/geofuncs.h"
#include "larflow/RecoUtils/cluster_t.h"

namespace larflow {
namespace recoutils {

  /**
   * @ingroup Reco
   * @class ClusterFunctions
   * @brief Empty class to help load the library
   * 
   */
  class ClusterFunctions {
    
  public:

    ClusterFunctions();
    virtual ~ClusterFunctions() {};
    
  };

  cluster_t cluster_from_larflowcluster( const larlite::larflowcluster& lfcluster );  
    
#ifndef __CINT__
#ifndef __CLING__
  
  void cluster_larflow3dhits( const std::vector<larlite::larflow3dhit>& hit_v,
                              std::vector< cluster_t >& cluster_v,
                              const float maxdist=5.0, const int minsize=5, const int maxkd=20 );

  void cluster_spacepoint_v( const std::vector< std::vector<float> >& hit_v,
                             std::vector< cluster_t >& cluster_v,
                             const float maxdist=5.0, const int minsize=5, const int maxkd=20 );
  
  void cluster_sdbscan_larflow3dhits( const std::vector<larlite::larflow3dhit>& hit_v,
                                      std::vector< cluster_t >& cluster_v,
                                      const float maxdist=5.0, const int minsize=5, const int maxkd=20 );
  
  void cluster_sdbscan_spacepoints( const std::vector< std::vector<float> >& hit_v,
                                    std::vector< cluster_t >& cluster_v,
                                    const float maxdist=5.0, const int minsize=5, const int maxkd=20 );
  

  /* not working */
  /* void cluster_dbscan_vp_larflow3dhits( const std::vector<larlite::larflow3dhit>& hit_v, */
  /*                                       std::vector< cluster_t >& cluster_v, */
  /*                                       const float maxdist=5.0, const int minsize=5, const int maxkd=5 ); */
  
  nlohmann::json cluster_json( const cluster_t& cluster );
  
  void cluster_dump2jsonfile( const std::vector<cluster_t>& cluster_v,
                              std::string outfilename );  

  void cluster_pca( cluster_t& cluster );

  void cluster_runpca( std::vector<cluster_t>& cluster_v );

  void cluster_splitbytrackshower( const std::vector<larlite::larflow3dhit>& hit_v,
                                   const std::vector<larcv::Image2D>& ssnettrack_image_v,
                                   std::vector<larlite::larflow3dhit>& track_hit_v,
                                   std::vector<larlite::larflow3dhit>& shower_hit_v,
                                   float min_larmatch_score=0.0 );

  void cluster_imageprojection( const cluster_t& cluster, std::vector<larcv::Image2D>& clust2d_images_v );

  void cluster_getcontours( std::vector<larcv::Image2D>& clust2d_images_v );

  float cluster_closest_endpt_dist( const cluster_t& clusta, const cluster_t& clust_b,
                                    std::vector< std::vector<float> >& endpts );

  bool cluster_endpt_in_bbox( const cluster_t& clust_a, const cluster_t& clust_b );
  
  float cluster_cospca( const cluster_t& clusta, const cluster_t& clustb );

  cluster_t cluster_merge( const cluster_t& clust_a, const cluster_t& clust_b );

  larlite::pcaxis cluster_make_pcaxis( const cluster_t& cluster, int id=0 );
 
  larlite::pcaxis cluster_make_pcaxis_wrt_point( const cluster_t& c,
                                                 const std::vector<float>& refpt,
                                                 int cidx=0 );
  
  std::vector<float> cluster_pixelsum( const cluster_t& cluster,
                                       const std::vector<larcv::Image2D>& img_v  );

  float cluster_dist_from_pcaline( const cluster_t& cluster,
                                   const std::vector<float>& pt );

  bool cluster_is_point_within_seg( const cluster_t& cluster,
				    const std::vector<float>& pt );
  
  void cluster_append( cluster_t& merge, const cluster_t& clust_b );

  void cluster_bbox( cluster_t& cluster );
    
  float cluster_dist_to_bbox( const cluster_t& cluster, const std::vector<float>& testpt  );

  int cluster_closest_pcaend( const cluster_t& cluster, const std::vector<float>& testpt );

  larlite::track cluster_make_trunk( const cluster_t& cluster, const std::vector<float>& vtxpos );

#endif
#endif

}
}

#endif
