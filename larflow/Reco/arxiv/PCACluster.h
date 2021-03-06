#ifndef __LARFLOW_PCA_CLUSTER_H__
#define __LARFLOW_PCA_CLUSTER_H__

#include <vector>
#include <string>

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflowcluster.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {
  
  class PCACluster {
  public:

    PCACluster()
      : _input_lfhit_tree_name("larmatch"),
      _min_larmatch_score(0.0),
      _downsample_fraction(1.0),
      _maxdist(10.0),
      _minsize(5),
      _maxkd(10)
      {};
    virtual ~PCACluster() {};

    void process( larcv::IOManager& iolc, larlite::storage_manager& ioll );

    void set_min_larmatch_score( float min_score ) { _min_larmatch_score = min_score; };
    void set_downsample_fraction( float keepfrac ) { _downsample_fraction = keepfrac; };
    void set_dbscan_pars( float maxdist, int minsize, int maxkd ) {
      _maxdist = maxdist;
      _minsize = minsize;
      _maxkd = maxkd;
    };

    static int split_clusters( std::vector<cluster_t>& cluster_v,
                               const std::vector<larcv::Image2D>& adc_v,
                               const float min_second_pca_len );

  protected:    
    
    int merge_clusters( std::vector<cluster_t>& cluster_v,
                        const std::vector<larcv::Image2D>& adc_v,
                        float max_dist_cm, float min_angle_deg, float max_pca2,
                        bool print_tests=false );

    void defragment_clusters( std::vector<cluster_t>& cluster_v,
                              const float max_2nd_pca_eigenvalue );

    void absorb_nearby_hits( cluster_t& cluster, std::vector<int>& used_hits_v );

    larlite::larflowcluster makeLArFlowCluster( cluster_t& cluster,
                                                const std::vector<larcv::Image2D>& ssnet_showerimg_v,
                                                const std::vector<larcv::Image2D>& ssnet_trackimg_v,
                                                const std::vector<larcv::Image2D>& adc_v,
                                                const std::vector<larlite::larflow3dhit>& source_lfhit_v );

    cluster_t absorb_nearby_hits( const cluster_t& cluster,
                                  const std::vector<larlite::larflow3dhit>& hit_v,
                                  std::vector<int>& used_hits_v,
                                  float max_dist2line );
    
    void multipassCluster( const std::vector<larlite::larflow3dhit>& inputhits,
                           const std::vector<larcv::Image2D>& adc_v,
                           std::vector<cluster_t>& output_cluster_v,
                           std::vector<int>& used_hits_v );

    std::string _input_lfhit_tree_name;
    float _min_larmatch_score;
    float _downsample_fraction;
    float _maxdist;
    int   _minsize;
    int   _maxkd;

  public:
     
    void set_input_larmatchhit_tree_name( std::string name ) { _input_lfhit_tree_name=name; };
    
  };

}
}

#endif
