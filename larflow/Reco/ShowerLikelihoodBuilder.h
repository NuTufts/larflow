#ifndef __LARFLOW_RECO_SHOWERLIKELIHOODBUILDER_H__
#define __LARFLOW_RECO_SHOWERLIKELIHOODBUILDER_H__


#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"
#include "larflow/Reco/cluster_functions.h"

#include "TFile.h"
#include "TH2F.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class ShowerLikelihoodBuilder
   * @brief Tools to build shower shape likelihood
   *
   * This class contains functions to build distibutions needed 
   * to make shower likelihood functions to be used in the larfow shower reco code
   *
   * This includes:
   * @verbatim embed:rst:leading-asterisk
   *  * shower profile likelihood in 3D. its the location of charge deposited as a function of 
   *       the distance along the trunk line and the perpendicular dist from the trunk line
   *  * brem segment impact param, distance to trunk line, cosine of pca between trunk lines
   * @endverbatim
   *
   * We fill a tree to later use to make distributions.
   * Can feed it, single shower MC  (best) or low energy neutrino. OK.
   *
   */  
  class ShowerLikelihoodBuilder {

  public:

    ShowerLikelihoodBuilder();
    virtual ~ShowerLikelihoodBuilder();

    std::string _wire_tree_name; ///< name of ROOT tree to look for
    /** @brief set name of tree to get wire plane data */
    void set_wire_tree_name( std::string name ) { _wire_tree_name = name; };
    
    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    void clear();

    larflow::prep::PrepMatchTriplets tripletalgo; ///< class that produces spacepoints from the wire plane images

    void _fillProfileHist( const std::vector<larlite::larflow3dhit>& truehit_v,
                           std::vector<float>& shower_dir,
                           std::vector<float>& shower_vtx );

    void _dist2line( const std::vector<float>& ray_start,
                     const std::vector<float>& ray_dir,
                     const std::vector<float>& pt,
                     float& radial_dist, float& projection );

    std::vector< cluster_t > cluster_v; ///< container for clusters of true shower hits
    
    void _make_truehit_clusters( std::vector< larlite::larflow3dhit >& truehit_v );

    int _find_closest_cluster( std::vector<int>& claimed_cluster_v,
                               std::vector<float>& shower_vtx,
                               std::vector<float>& shower_dir );
    

    int _trunk_cluster; ///< index of cluster that is the trunk
    std::vector< float >     cluster_pcacos2trunk_v; ///< cosine between first PC of sub-cluster and trunk cluster of a true shower
    std::vector< float >     cluster_dist2trunk_v;   ///< distance between sub-cluster and trunk-cluster
    std::vector< float >     cluster_impactdist2trunk_v; ///< impact parameter between sub-cluster and trunk-cluster
   
    void _analyze_clusters( std::vector< larlite::larflow3dhit >& truehit_v,
                            std::vector<float>& shower_dir,
                            std::vector<float>& shower_vtx );
    

    larutil::SpaceChargeMicroBooNE* _psce; ///< pointer to a copy of the space charge offset calculating algo
    TH2F* _hll;  ///< 2D likelihood distribution over distance along and perpendicular to the trunk's first principle component
    TH2F* _hll_weighted; ///< likelihood weighed by R^2, the squared-distnce perpendicular from the trunk
    TTree* _tree_cluster_relationships; ///< tree containing observables between sub-cluster and trunk-cluster. used to build selection method
    float _dist2trunk; ///< distance of true vertex to the trunk
    
    TTree* _tree_trunk_features; ///< tree containing features that characterize the shower-trunk cluster, to help select trunk from sub-clusters

    void _impactdist( const std::vector<float>& l_start,
                      const std::vector<float>& l_dir,
                      const std::vector<float>& m_start,
                      const std::vector<float>& m_dir,
                      float& impact_dist,
                      float& proj_l, float& proj_m );

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
    };
    std::vector<ShowerInfo_t> _shower_info_v; ///< vector of info on true shower objects in event
    std::vector< larlite::larflowcluster > _larflow_cluster_v; ///< larflow cluster made by merged true hit clusters based on trunk info from shower objects
    bool _kExcludeCosmicShowers;
    void set_exclude_cosmic_showers( bool exclude ) { _kExcludeCosmicShowers=exclude; };

    void _make_shower_info( const larlite::event_mcshower& ev_mcshower,
                            std::vector< ShowerInfo_t>& info_v,
                            bool exclude_cosmic_showers );
    
    void _trueshowers_absorb_clusters( std::vector<ShowerInfo_t>& shower_info_v,
                                       std::vector<larlite::larflowcluster>& merged_cluster_v,
                                       const std::vector<larlite::larflow3dhit>& truehit_v );

    void updateMCPixelGraph( ublarcvapp::mctools::MCPixelPGraph& mcpg,
                             larcv::IOManager& iolcv ); 
    
    
  };
  
}
}

#endif
