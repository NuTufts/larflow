#ifndef __LARFLOW_RECO_SHOWERLIKELIHOODBUILDER_H__
#define __LARFLOW_RECO_SHOWERLIKELIHOODBUILDER_H__


#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"

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

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

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
    
  };
  
}
}

#endif
