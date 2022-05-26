#ifndef __LARFLOW_RECO_KEYPOINTRECO_H__
#define __LARFLOW_RECO_KEYPOINTRECO_H__

#include <vector>

#include "TTree.h"

// larcv
#include "larcv/core/Base/larcv_base.h"

// larlite
#include "larlite/DataFormat/larflow3dhit.h"
#include "larlite/DataFormat/storage_manager.h"

#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "larflow/Reco/cluster_functions.h"
#include "larflow/Reco/KPCluster.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class KeypointReco
   * @brief Produce keypoints from scores made by larmatch keypoint network
   *
   */
  class KeypointReco : public larcv::larcv_base {

  public:

    /** @brief default constructor */
    KeypointReco()
      : larcv::larcv_base("KeypointReco"),
      _output_tree(nullptr)
    { set_param_defaults(); };
    virtual ~KeypointReco()
      {
        if (_output_tree) {
          delete _output_tree;
          _output_tree = nullptr;
        }
        
      };

    // params
  protected:

    float _sigma;                                   ///< width of keypoint. used to supress neighbor keypoints
    float _larmatch_score_threshold;                ///< ignore spacepoints with larmatch score below this values    
    int   _num_passes;                              ///< number of passes to perform
    std::vector<float> _keypoint_score_threshold_v; ///< ignore spacepoints with keypoint score below this values, threshold for each pass    
    std::vector<int>   _min_cluster_size_v;         ///< minimum cluster size to be a vertex (cluster of hits above threshold)
    float _max_dbscan_dist;                         ///< max distance parameter in DBscan used
    int   _max_clustering_points;                   ///< maximum number of points to use when clustering. if have more, sampled randomly.
    std::string _input_larflowhit_tree_name;        ///< name of the input container tree with larflow hits to use
    std::string _output_tree_name;                  ///< name of the output container tree to put keypoints in
    int   _keypoint_type;                           ///< label of keypoint type we're making
    int   _lfhit_score_index;                       ///< index of column in larflow3d hit info vector with keypoint score
    float _threshold_cluster_max_score;             ///< the threshold max keypoint score in a keypoint cluster
    int   _min_kpcluster_size;                      ///< min size of keypoint cluster
    
  public:
    
    void set_param_defaults();

    /** @brief set keypoint threshold score for a given pass */
    void set_keypoint_threshold( float threshold, int pass=0 )    { _keypoint_score_threshold_v[pass] = threshold; };

    /** @brief set larmatch threshol score */    
    void set_larmatch_threshold( float threshold )    { _larmatch_score_threshold=threshold; };

    /** @brief set sigma dictating how fast score drops off */
    void set_sigma( float sigma )            { _sigma=sigma; };

    /** @brief set minimum cluster size when clustering around keypoint */
    void set_min_cluster_size( int minsize, int pass=0 ) { _min_cluster_size_v[pass] = minsize;  };

    /** @brief set number of passes */
    void set_num_passes( int npasses )       {
      _num_passes = npasses;
      _keypoint_score_threshold_v.resize(npasses,0.5);
      _min_cluster_size_v.resize(npasses,50);
    };

    /** @brief set maximum distance two spacepoints can be before not being connected */
    void set_max_dbscan_dist( float dist )   { _max_dbscan_dist = dist; };

    /** @brief set maximum nunmber of clustering points */
    void set_max_clustering_points( int maxpts ) { _max_clustering_points = maxpts; };

    /** @brief set keypoint type we are reconstucting with this class */
    void set_keypoint_type (int kptype ) { _keypoint_type=kptype; };

    /** @brief set name of tree with larmatch larflow3dhits */
    void set_input_larmatch_tree_name( std::string name ) { _input_larflowhit_tree_name=name; };

    /** @brief set output tree name with keypoints stored as larflow3dhits */
    void set_output_tree_name( std::string name ) { _output_tree_name=name; };

    /** @brief set index of feature vector of larflow3dhit where larmatch score is stored */
    void set_lfhit_score_index( int idx ) { _lfhit_score_index=idx; };
    

    void process( larlite::storage_manager& io_ll );
    void process_all_keypoint_types( larlite::storage_manager& io_ll );    
    void process( const std::vector<larlite::larflow3dhit>& input_lfhits );
    void dump2json( std::string outfilename="dump_keypointreco.json" );    

    std::vector< KPCluster > output_pt_v;  ///< container of class representing reco. vertices
    std::vector< cluster_t >   _cluster_v; ///< clusters of spacepoints with keypoint score above some threshold

    // make initial points
    std::vector< std::vector<float> > _initial_pt_pos_v;  ///< initial points we are searching (x,y,z,kp-score,lm-score)
    std::vector< int >                _initial_pt_used_v; ///< flag indicating point is used
    
    void _make_initial_pt_data( const std::vector<larlite::larflow3dhit>& lfhits,
                                const float keypoint_score_threshold,
                                const float larmatch_score_threshold,
				const int tpcid, const int cryoid );

    void _make_kpclusters( float round_score_threshold, int min_cluster_size, const int tpcid, const int cryoid );
    
    // algorithm steps
    void _skim_remaining_points( float score_threshold,
                                 std::vector<std::vector<float> >& skimmed_pt_v,
                                 std::vector<int>& skimmed_index_v );

    KPCluster _characterize_cluster( cluster_t& cluster,
                                     std::vector< std::vector<float> >& skimmed_pt_v,
                                     std::vector< int >& skimmed_index_v );

    KPCluster _fit_cluster_CARUANA( cluster_t& cluster,
				    std::vector< std::vector<float> >& skimmed_pt_v,
				    std::vector< int >& skimmed_index_v );
    
    
    void _expand_kpcluster( KPCluster& kp );

    void printAllKPClusterInfo();

  protected:

    TTree* _output_tree; ///< point to tree where output is saved. optional.

  public:

    void bindKPClusterContainerToTree( TTree* out );    
    void setupOwnTree();

    /** @brief write _output_tree to ROOT file if defined */
    void writeTree() { if ( _output_tree ) _output_tree->Write(); };

    /** @brief fill entry data in _output_tree */
    void fillTree() {  if ( _output_tree ) _output_tree->Fill(); };

  };

}
}

#endif
