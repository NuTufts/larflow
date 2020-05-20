#ifndef __LARFLOW_DBSCAN_LARMATCH_H__
#define __LARFLOW_DBSCAN_LARMATCH_H__

#include <vector>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflowcluster.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {
  
  class DBScanLArMatchHits : public larcv::larcv_base {
  public:

    DBScanLArMatchHits()
      : _input_larflowhit_tree_name("larmatch"),
      _out_cluster_tree_name("dblarmatch"),
      _min_larmatch_score(0.0),
      _downsample_fraction(1.0),
      _maxdist(10.0),
      _minsize(5),
      _maxkd(10)
      {};
    virtual ~DBScanLArMatchHits() {};

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

    void absorb_nearby_hits( cluster_t& cluster, std::vector<int>& used_hits_v );

    larlite::larflowcluster makeLArFlowCluster( cluster_t& cluster,
                                                const std::vector<larlite::larflow3dhit>& source_lfhit_v );

    cluster_t absorb_nearby_hits( const cluster_t& cluster,
                                  const std::vector<larlite::larflow3dhit>& hit_v,
                                  std::vector<int>& used_hits_v,
                                  float max_dist2line );
    
    void makeCluster( const std::vector<larlite::larflow3dhit>& inputhits,
                      std::vector<cluster_t>& output_cluster_v,
                      std::vector<int>& used_hits_v );

  protected:
    
    // PARAMETERS

    std::string _input_larflowhit_tree_name;
    std::string _out_cluster_tree_name;
    float _min_larmatch_score;
    float _downsample_fraction;
    float _maxdist;
    int   _minsize;
    int   _maxkd;

  public:

    void set_input_larflowhit_tree_name( std::string treename )
    { _input_larflowhit_tree_name=treename; };

    void set_output_larflowhit_tree_name( std::string treename )
    { _out_cluster_tree_name = treename; };
    
  };

}
}

#endif
