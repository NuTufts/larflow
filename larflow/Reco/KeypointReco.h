#ifndef __LARFLOW_RECO_KEYPOINTRECO_H__
#define __LARFLOW_RECO_KEYPOINTRECO_H__

#include <vector>

// larlite
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/storage_manager.h"

#include "cluster_functions.h"
#include "KPCluster.h"

namespace larflow {
namespace reco {
  
  class KeypointReco  {

  public:
    
    KeypointReco() { set_param_defaults(); };
    virtual ~KeypointReco() {};

    // params
  protected:

    float _sigma;            //< width of keypoint. used to supress neighbor keypoints
    float _score_threshold;  //< ignore spacepoints with keypoint score below this values
    int   _num_passes;       //< number of passes to perform
    int   _min_cluster_size; //< minimum cluster size to be a vertex (cluster of hits above threshold)
    float _max_dbscan_dist;  //< max distance parameter in DBscan used
  public:
    void set_param_defaults();
    void set_threshold( float threshold )    { _score_threshold=threshold; };
    void set_sigma( float sigma )            { _sigma=sigma; };
    void set_min_cluster_size( int minsize ) { _min_cluster_size=minsize; };
    void set_num_passes( int npasses )       { _num_passes = npasses; };
    void set_max_dbscan_dist( float dist )   { _max_dbscan_dist = dist; };
    

    void process( larlite::storage_manager& io_ll );
    void process( const std::vector<larlite::larflow3dhit>& input_lfhits );
    void dump2json( std::string outfilename="dump_keypointreco.json" );    

    std::vector< KPCluster > output_pt_v;
    std::vector< cluster_t >   _cluster_v;

    // make initial points
    std::vector< std::vector<float> > _initial_pt_pos_v;  ///< initial points we are searching (x,y,z,score)
    std::vector< int >                _initial_pt_used_v; ///< flag indicating point is used
    void _make_initial_pt_data( const std::vector<larlite::larflow3dhit>& lfhits,
                                float score_threshold );

    void _make_kpclusters( float round_score_threshold );
    
    // algorithm steps
    void _skim_remaining_points( float score_threshold,
                                 std::vector<std::vector<float> >& skimmed_pt_v,
                                 std::vector<int>& skimmed_index_v );

    KPCluster _characterize_cluster( cluster_t& cluster,
                                     std::vector< std::vector<float> >& skimmed_pt_v,
                                     std::vector< int >& skimmed_index_v );
    
    void _expand_kpcluster( KPCluster& kp );

    void printAllKPClusterInfo();

  };

}
}

#endif
