#ifndef __LARFLOW_RECO_NU_TRACK_KINEMATICS_H__
#define __LARFLOW_RECO_NU_TRACK_KINEMATICS_H__

#include "TLorentzVector.h"
#include "TFile.h"
#include "TSpline.h"
#include "larcv/core/Base/larcv_base.h"
#include "DataFormat/track.h"

#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"

namespace larflow {
namespace reco {

  class NuTrackKinematics : public larcv::larcv_base {

  public:
    
    NuTrackKinematics();
    virtual ~NuTrackKinematics();

    std::vector<float>          _track_length_v;  ///< track lengths
    std::vector<float>          _track_mu_ke_v;   ///< track KE assuming muon
    std::vector<float>          _track_p_ke_v;    ///< track KE assuming proton
    std::vector<TLorentzVector> _track_mu_mom_v;  ///< track momentum assuming muon
    std::vector<TLorentzVector> _track_p_mom_v;   ///< track momentum assuming proton

    /* std::vector<TLorentzVector> _track_ll_v;         ///< proton versus muon likelihood */
    /* std::vector<TLorentzVector> _track_bestll_mom_v; ///< momentum using best likelihood */

    void clear();
    void analyze( larflow::reco::NuVertexCandidate& nuvtx );
      
  protected:
    
    TFile*    _splinefile_rootfile;
    TSpline3* _sMuonRange2T;
    TSpline3* _sProtonRange2T;

    void _load_data();
    float get_tracklen( const larlite::track& track );
    TVector3 get_trackdir_radius( const larlite::track& track,
                                  const float radius,
                                  const std::vector<float>& vtx );
    


  };
  
}
}

#endif
