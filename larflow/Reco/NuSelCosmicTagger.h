#ifndef __LARFLOW_RECO_NUSEL_COSMIC_TAGGER_H__
#define __LARFLOW_RECO_NUSEL_COSMIC_TAGGER_H__

#include "TVector3.h"
#include "larlite/DataFormat/track.h"
#include "larcv/core/Base/larcv_base.h"
#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"

namespace larflow {
namespace reco {

  class NuSelCosmicTagger : public larcv::larcv_base {

  public:

    NuSelCosmicTagger()
      : larcv::larcv_base("NuSelCosmicTagger")
      {};

    virtual ~NuSelCosmicTagger() {};

    void analyze( larflow::reco::NuVertexCandidate& nuvtx,
                  larflow::reco::NuSelectionVariables& nusel );

    void tagShoweringMuon2Track( larflow::reco::NuVertexCandidate& nuvtx,
                                 larflow::reco::NuSelectionVariables& nusel );
    
    void tagShoweringMuon1Track( larflow::reco::NuVertexCandidate& nuvtx,
                                 larflow::reco::NuSelectionVariables& nusel );
    
    void tagStoppingMuon( larflow::reco::NuVertexCandidate& nuvtx,
                          larflow::reco::NuSelectionVariables& nusel );

    // 2-track showering muon tagger
    float _showercosmictag_mindwall_dwall;
    float _showercosmictag_mindwall_costrack;
    float _showercosmictag_maxbacktoback_dwall;
    float _showercosmictag_maxbacktoback_costrack;
    
    // 1-track showering muon tagger
    float _showercosmictag_maxboundarytrack_length;
    float _showercosmictag_maxboundarytrack_verticalcos;
    float _showercosmictag_maxboundarytrack_showercos;
    
    
  protected:
    
    TVector3 _defineTrackDirection( const larlite::track& track );
    float _getTrackLength( const larlite::track& track );
    
  };
  
}
}

#endif
