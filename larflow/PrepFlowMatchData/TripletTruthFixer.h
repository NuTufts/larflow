#ifndef __LARFLOW_PREP_TRIPLETTRUTHFIXER_H__
#define __LARFLOW_PREP_TRIPLETTRUTHFIXER_H__

#include "core/DataFormat/storage_manager.h"
#include "core/DataFormat/mcshower.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "larflow/Reco/cluster_functions.h"
#include "PrepMatchTriplets.h"

namespace larflow {
namespace prep {

  /**
   * @ingroup PrepFlowMatchData
   * @class TripletTruthFixer
   * @brief Uses clustering and larlite truth to repair triplet spacepoint instance labels
   *
   */
  class TripletTruthFixer {

  public:

    TripletTruthFixer()
      : _kExcludeCosmicShowers(true)
      {};
    virtual ~TripletTruthFixer() {};
    
    void calc_reassignments( PrepMatchTriplets& tripmaker,
                             larcv::IOManager& iolcv,
                             larlite::storage_manager& ioll );

    void _cluster_same_showerpid_spacepoints( std::vector<larflow::reco::cluster_t>& cluster_v,
                                              std::vector<int>& pid_v,
                                              std::vector<int>& shower_instance_v,
                                              larflow::prep::PrepMatchTriplets& tripmaker,
                                              bool reassign_instance_labels );

    void _reassignSmallTrackClusters( larflow::prep::PrepMatchTriplets& tripmaker,
                                      const std::vector< larcv::Image2D >& instanceimg_v,
                                      const float threshold );

    void _merge_shower_fragments( std::vector<larflow::reco::cluster_t>& shower_fragments_v,
                                  std::vector<int>& pid_v,
                                  std::vector<int>& shower_instance_v,
                                  std::vector<larflow::reco::cluster_t>& merged_showers_v );

    /**
     * @struct ShowerInfo_t
     * @brief Info we are tracking with true shower objects
     * 
     */
    struct ShowerInfo_t {
      int idx; ///< index in mc shower event container
      int trackid; ///< track id from geant for particle
      int pid; ///< particle ID
      int origin; ///< origin index
      int priority; ///< rank of shower when absorbing clusters
      int matched_cluster; ///< cluster that matches to start point
      float highq_plane; ///< charge of shower on the highest plane, no idea what the unit is
      float cos_sce; ///< direction difference between det profile dir before and after SCE
      float E_MeV; ///< energy of starting lepton (use to set maximum shower distance)
      std::vector<float> shower_dir; ///< det profile direction
      //std::vector<float> shower_dir_sce;
      std::vector<float> shower_vtx; ///< det profile shower start
      //std::vector<float> shower_vtx_sce;
      /** @brief comparison operator for struct used for sorting by priority and charge */
      bool operator<(const ShowerInfo_t& rhs ) {
        if ( priority<rhs.priority ) return true;
        else if ( priority==rhs.priority && highq_plane>rhs.highq_plane ) return true;
        return false;
      };
      std::vector<int> absorbed_cluster_index_v; ///< indices of clusters absorbed
      ShowerInfo_t()
      {
        absorbed_cluster_index_v.clear();
      };
    };

  protected:
    
    std::vector<ShowerInfo_t> _shower_info_v; ///< vector of info on true shower objects in event
    bool _kExcludeCosmicShowers; ///< if true, ignore shower clusters
    void _make_shower_info( const larlite::event_mcshower& ev_mcshower,
                            std::vector< ShowerInfo_t>& info_v,
                            bool exclude_cosmic_showers );
    

    int _find_closest_cluster( std::vector< larflow::reco::cluster_t >& shower_fragment_v,
                               std::vector<int>& claimed_cluster_v,
                               std::vector<float>& shower_vtx,
                               std::vector<float>& shower_dir );

    void _trueshowers_absorb_clusters( std::vector<ShowerInfo_t>& shower_info_v,
                                       std::vector<larflow::reco::cluster_t>& shower_fragment_v,
                                       std::vector<int>& fragment_pid_v,                                       
                                       std::vector<int>& cluster_used_v,
                                       std::vector<larflow::reco::cluster_t>& merged_cluster_v );
    
    void _reassign_merged_shower_instance_labels( std::vector<larflow::reco::cluster_t>& merged_showers_v,
                                                  std::vector<ShowerInfo_t>& shower_info_v,
                                                  larflow::prep::PrepMatchTriplets& tripmaker );
    
    
    
  };

}
}

#endif
