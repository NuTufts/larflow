#ifndef __LARFLOW_RECO_NUSEL_SHOWERTRUNK_ANA_H__
#define __LARFLOW_RECO_NUSEL_SHOWERTRUNK_ANA_H__

#include <vector>

#include "DataFormat/storage_manager.h"
#include "DataFormat/track.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"
#include "ShowerdQdx.h"

namespace larflow {
namespace reco {

  class NuSelShowerTrunkAna : public larcv::larcv_base {

  public:

    NuSelShowerTrunkAna()
      : larcv::larcv_base("NuSelShowerTrunkAna")
      {};

    virtual ~NuSelShowerTrunkAna() {};

    void analyze( larflow::reco::NuVertexCandidate& nuvtx,
                  larflow::reco::NuSelectionVariables& output,
                  larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll );    
    
    std::vector<larlite::track>       _shower_dqdx_v;     ///< for each reco shower, dq/dx values along shower trunk
    std::vector< std::vector<float> > _shower_avedqdx_v;  ///< for each reco shower, dq/dx per plane
    std::vector< std::vector<float> > _shower_ll_v;       ///< for each reco shower, likelihood per plane
    std::vector<float>                _shower_gapdist_v;  ///< for each reco shower, distance from shower start to vertex
    
    std::vector< int >                _shower_true_match_index_v; ///< index in mcshower container of best matching true shower
    std::vector< int >                _shower_true_match_pdg_v;   ///< PDG code of best matching true shower
    std::vector< float >              _shower_true_match_cos_v;   ///< cosine between best matching shower and reco shower trunk directions
    std::vector< float >              _shower_true_match_vtx_err_dist_v; ///< distance between true vertex and reco shower start, projected onto shower direction

    larflow::reco::ShowerdQdx dqdx_algo; ///< code to calculate shower dqdx

    bool refineShowerTrunk( const larlite::track& shower_trunk,
                            const larlite::pcaxis& shower_pca,
                            const TVector3& tvtx,
                            std::vector<float>& fstart,
                            std::vector<float>& fend,
                            std::vector<float>& fdir,
                            float& gapdist );

    void clear();
    void storeShowerRecoSentinelValues();
    void storeShowerTruthSentinelValues();
    
  };
  
}
}


#endif
