#ifndef __LARFLOW_RECO_KEYPOINTRECO_H__
#define __LARFLOW_RECO_KEYPOINTRECO_H__

#include <vector>

// larlite
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/storage_manager.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {

  struct KPCluster_t {
    
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

    int _cluster_idx;                             ///< associated cluster_t in KeypointReco::_cluster_v (internal use only)
    //cluster_t cluster;
  };
  
  class KeypointReco  {

  public:
    
    KeypointReco() {};
    virtual ~KeypointReco() {};

    void process( larlite::storage_manager& io_ll );
    void process( const std::vector<larlite::larflow3dhit>& input_lfhits );
    void dump2json( std::string outfilename="dump_keypointreco.json" );    

    std::vector< KPCluster_t > output_pt_v;
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

    KPCluster_t _characterize_cluster( cluster_t& cluster,
                                       std::vector< std::vector<float> >& skimmed_pt_v,
                                       std::vector< int >& skimmed_index_v );
    
    void _expand_kpcluster( KPCluster_t& kp );

  };

}
}

#endif
