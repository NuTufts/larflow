#ifndef __LARFLOW_SHOWER_RECO_KEYPOINT_H__
#define __LARFLOW_SHOWER_RECO_KEYPOINT_H__

#include <vector>
#include <set>

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/larflowcluster.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {

  /** 
   * @ingroup Reco
   * @class ShowerRecoKeypoint
   * @brief Reconstruct a shower using shower-labeled clusters
   *
   */
  class ShowerRecoKeypoint : public larcv::larcv_base {

  public:

    ShowerRecoKeypoint()
      : larcv::larcv_base("ShowerRecoKeypoint"),
      _ssnet_lfhit_tree_name("showerhit"),
      _larmatch_score_threshold(0.5),
      _shower_rad_threshold_cm(3.0)
      {};    
    virtual ~ShowerRecoKeypoint() {};
    
    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    void setShowerRadiusThresholdcm( float thresh ) { _shower_rad_threshold_cm=thresh; };    

  protected:

    /**
     * @struct ShowerTrunk_t
     * @brief internal type representing a shower trunk
     */
    struct ShowerTrunk_t {
      int idx_keypoint; ///< index of keypoint in the event container
      const larlite::larflow3dhit* keypoint; ///< pointer to keypoint object used to seed this trunk
      std::vector< float > pcaxis_v; ///< first principle component of cluster
      std::vector< float > center_v; ///< centroid of cluster
      std::vector< float > start_v;  ///< start point of trunk
      float pca_eigenval_ratio;      ///< ratio of second to first principle component eigenvalue
      int npts;                      ///< number of spacepoints in trunk cluster
      float gapdist;                 ///< distance from keypoint
      float impact_par;              ///< distance of first principle component to keypoint
      ShowerTrunk_t() {
        idx_keypoint = -1;
        keypoint = nullptr;        
        pcaxis_v.resize(3,0);
        center_v.resize(3,0);
        start_v.resize(3,0);
        pca_eigenval_ratio = -1;
        npts = -1;
        gapdist = -1;
        impact_par = -1;
      };
    };

    /**
     * @struct ShowerCandidate_t
     * @brief internal type representing a cluster and trunks that might be associated to it.
     */
    struct ShowerCandidate_t {
      int cluster_idx; ///< index of shower cluster
      const cluster_t* cluster; ///< pointer to shower cluster
      std::vector< ShowerTrunk_t > trunk_candidates_v; ///< possible trunks that this cluster is a part of
    };

    /**
     * @struct Shower_t
     * @brief represents final shower reconstructed object
     */
    struct Shower_t  {
      ShowerTrunk_t trunk; ///< trunk info
      std::set<int> cluster_idx; ///< set of shower cluster idx
      std::vector< std::vector<float> > points_v; ///< 3d points assigned to shower
      std::vector< int > hitidx_v; ///< hit idx of 3d hits corresponding to input larflow3dhit vector
    };

    std::vector< ShowerCandidate_t > _shower_cand_v;  ///< collection of shower (sub)clusters+trunk forming a shower candidate
    std::vector< Shower_t >          _recod_shower_v; ///< final set of reconstructed showers
    
    void _reconstructClusterTrunks( const std::vector<const cluster_t*>&    showercluster_v,
                                    const std::vector<const larlite::larflow3dhit*>& keypoint_v );    
    void _buildShowers( const std::vector<const cluster_t*>&  showerhit_cluster_v );
    Shower_t _buildShowerCandidate( const ShowerCandidate_t& shower_cand,
                                    const std::vector<const cluster_t*>& showerhit_cluster_v );
    std::set<int> _buildoutShowerTrunkCandidate( const ShowerTrunk_t& trunk_cand,
                                                 const std::vector<const cluster_t*>& showerhit_cluster_v );

    Shower_t _fillShowerObject( const ShowerCandidate_t& shower_cand,
                                const std::set<int>& cluster_idx_set,
                                const int trunk_idx,
                                const std::vector< const cluster_t* >& showerhit_cluster_v );    

    void _fillShowerObject( Shower_t& shower,
                            const std::vector< const cluster_t* >& showerhit_cluster_v );

    int _chooseBestTrunk( const ShowerCandidate_t& shower_cand,
                          const std::set<int>& cluster_idx_v,
                          const std::vector< const cluster_t* >& showerhit_cluster_v );

    int _chooseBestShowerForCluster( const cluster_t& cluster,
                                     const std::set<int>& shower_idx_v,
                                     const std::vector< const cluster_t* >& showerhit_cluster_v );

    
  protected:
    
    // PARAMETERS
    std::string _ssnet_lfhit_tree_name; ///< name of tree to get input larflow3dhit
    float _larmatch_score_threshold;    ///< threshold of larmatch score to use larflow3dhit
    float _shower_rad_threshold_cm;        ///< radius from trunk to absorb clusters
    
  public:

    /** @brief set name of input larflow3dhit tree to use */
    void set_ssnet_lfhit_tree_name( std::string name ) { _ssnet_lfhit_tree_name=name; };

    /** @brief set larmatch score threshold */
    void set_larmatch_score_threshold( float thresh ) { _larmatch_score_threshold = thresh; };
    
    
  };
  

}
}

#endif
