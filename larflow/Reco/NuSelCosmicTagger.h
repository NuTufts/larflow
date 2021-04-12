#ifndef __LARFLOW_RECO_NUSEL_COSMIC_TAGGER_H__
#define __LARFLOW_RECO_NUSEL_COSMIC_TAGGER_H__

#include "TVector3.h"
#include "DataFormat/track.h"
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

    void tagShoweringMuon( larflow::reco::NuVertexCandidate& nuvtx,
                           larflow::reco::NuSelectionVariables& nusel );

    void tagStoppingMuon( larflow::reco::NuVertexCandidate& nuvtx,
                          larflow::reco::NuSelectionVariables& nusel );
    
  protected:
    
    TVector3 _defineTrackDirection( const larlite::track& track );

  };
  
}
}

#endif
