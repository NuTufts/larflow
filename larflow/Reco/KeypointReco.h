#ifndef __LARFLOW_RECO_KEYPOINTRECO_H__
#define __LARFLOW_RECO_KEYPOINTRECO_H__

#include <vector>

// larlite
#include "DataFormat/larflow3dhit.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {

  struct KPCluster_t {
    
    std::vector< float > center_pt_v;             ///< center point
    std::vector< std::vector<float> > pt_pos_v;   ///< points associated to the center
    std::vector< float >              pt_score_v; ///< score of cluster points
    cluster_t cluster;
  };
  
  class KeypointReco  {

  public:
    
    KeypointReco() {};
    virtual ~KeypointReco() {};

    void process( const std::vector<larlite::larflow3dhit>& input_lfhits );
    void dump2json( std::string outfilename="dump_keypointreco.json" );    

    std::vector< KPCluster_t > output_pt_v;

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
                                       std::vector< std::vector<float> >& skimmed_pt_v );
    
    /* void _absorb_points_into_clusters(); */

  };

}
}

#endif
