#ifndef __LARFLOW_PROJECTION_DEFECT_SPLITTER_H__
#define __LARFLOW_PROJECTION_DEFECT_SPLITTER_H__

#include <vector>
#include <string>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflowcluster.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {
  
  class ProjectionDefectSplitter : public larcv::larcv_base {
    
  public:

    ProjectionDefectSplitter()
      : larcv::larcv_base("ProjectionDefectSplitter"),
      _input_lfhit_tree_name("larmatch"),
      _output_cluster_tree_name("projsplit"),
      _min_larmatch_score(0.0),
      _downsample_fraction(1.0),
      _maxdist(2.0),
      _minsize(20),
      _maxkd(30)
      {};
    virtual ~ProjectionDefectSplitter() {};

    void process( larcv::IOManager& iolc, larlite::storage_manager& ioll );

    int split_clusters( std::vector<cluster_t>& cluster_v,
                        const std::vector<larcv::Image2D>& adc_v,
                        const float min_second_pca_len );

  protected:

    larlite::larflowcluster
      _makeLArFlowCluster( cluster_t& cluster,
                           const larlite::event_larflow3dhit& source_lfhit_v );
    
    cluster_t _absorb_nearby_hits( const cluster_t& cluster,
                                   const std::vector<larlite::larflow3dhit>& hit_v,
                                   std::vector<int>& used_hits_v,
                                   float max_dist2line );
    
    void _runSplitter( const larlite::event_larflow3dhit& inputhits,
                       const std::vector<larcv::Image2D>& adc_v,
                       std::vector<int>& used_hits_v,
                       std::vector<cluster_t>& output_cluster_v );

    void _defragment_clusters( std::vector<cluster_t>& cluster_v,
                               const float max_2nd_pca_eigenvalue );
    
    // PARAMETER NAMES
  protected:
    
    std::string _input_lfhit_tree_name;
    std::string _output_cluster_tree_name;
    float _min_larmatch_score;
    float _downsample_fraction;
    float _maxdist;
    int   _minsize;
    int   _maxkd;

  public:
     
    void set_input_larmatchhit_tree_name( std::string name ) { _input_lfhit_tree_name=name; };
    void set_output_tree_name( std::string name ) { _output_cluster_tree_name=name; };    
    void set_min_larmatch_score( float min_score ) { _min_larmatch_score = min_score; };
    void set_downsample_fraction( float keepfrac ) { _downsample_fraction = keepfrac; };
    void set_dbscan_pars( float maxdist, int minsize, int maxkd ) {
      _maxdist = maxdist;
      _minsize = minsize;
      _maxkd = maxkd;
    };
    
  };

}
}

#endif
