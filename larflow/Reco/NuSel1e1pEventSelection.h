#ifndef __LARFLOW_REFO_NUSEL_1E1P_EVENT_SELECTION_H__
#define __LARFLOW_REFO_NUSEL_1E1P_EVENT_SELECTION_H__

#include "larcv/core/Base/larcv_base.h"
#include "larflow/Reco/NuSelectionVariables.h"
#include "larflow/Reco/NuVertexCandidate.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class NuSel1e1pEventSelection
   * @brief Implements test 1e1p selection for development. Allows filter of vertices during Reco.
   *
   */
  class NuSel1e1pEventSelection : public larcv::larcv_base {
  public:

    NuSel1e1pEventSelection()
      : larcv::larcv_base("NuSel1e1pEventSelection")
      {};

    virtual ~NuSel1e1pEventSelection() {};

    int runSelection( const larflow::reco::NuSelectionVariables& nusel,
                      const larflow::reco::NuVertexCandidate& nuvtx ); ///< run selection
    

    // cut stages
    typedef enum { kMinShowerSize=0, // [0] min shower size cut (might want to loosen)
                   kNShowerProngs,   // [1] number of shower prongs
                   kNTrackProngs,    // [2] number of track prongs         
                   kShowerGap,       // [3] shower gap
                   kTrackGap,        // [4] track gap
                   kMaxTrackLen,     // [5] max track len
                   kSecondShower,    // [6] second shower size
                   kVertexAct,       // [7] vertex activity cut
                   kRecoFV,          // [8] reco fv cut
                   kShowerLLCut,     // [9] shower likelihood cut         
                   kWCPixel,         // [10] Wire-Cell pixel cut
                   kHadronic,        // [11] see hadronic particles (proton or vertex activity)         
                   kAllCuts,         // [12] All cuts applied except FV -- represents reco pass rate
                   kNumCuts }        // [13] Number in enum
    CutStages_t;
    
  };
  
}
}

#endif
