#ifndef __LARFLOW_RECO_SHOWERLIKELIHOODBUILDER_H__
#define __LARFLOW_RECO_SHOWERLIKELIHOODBUILDER_H__


#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/larflow3dhit.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"
#include "larflow/Reco/cluster_functions.h"

#include "TFile.h"
#include "TH2F.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup MCTools
   * @class ShowerLikelihoodBuilder
   * @brief Tools to build shower shape likelihood
   *
   * This class contains functions to build distibutions needed 
   * to make shower likelihood functions to be used in the larfow shower reco code.
   * Expects true spacepoint data built using larflow::prep::PrepMatchTriplets
   * and larflow::prep::TripletTruthFixer.
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

    void clear();

    void _fillProfileHist( const std::vector<larlite::larflow3dhit>& truehit_v,
                           std::vector<float>& shower_dir,
                           std::vector<float>& shower_vtx );

    
    larutil::SpaceChargeMicroBooNE* _psce; ///< pointer to a copy of the space charge offset calculating algo
    TH2F* _hll;  ///< 2D likelihood distribution over distance along and perpendicular to the trunk's first principle component
    TH2F* _hll_weighted; ///< likelihood weighed by R^2, the squared-distnce perpendicular from the trunk
    TTree* _tree_cluster_relationships; ///< tree containing observables between sub-cluster and trunk-cluster. used to build selection method
    float _dist2trunk; ///< distance of true vertex to the trunk
    
    TTree* _tree_trunk_features; ///< tree containing features that characterize the shower-trunk cluster, to help select trunk from sub-clusters

    
  };
  
}
}

#endif
