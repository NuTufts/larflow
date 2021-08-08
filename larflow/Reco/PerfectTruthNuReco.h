#ifndef __LARFLOW_RECO_PERFECT_TRUTH_NU_RECO_H__
#define __LARFLOW_RECO_PERFECT_TRUTH_NU_RECO_H__

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/larflow3dhit.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"

#include "NuVertexCandidate.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class PerfectTruthNuReco
   * @brief Uses true shower and track trajectories + (reco) larmatch points to make NuVertexCandidate
   *
   * The output is meant to have a way to evaluate performance of reco.
   * Things we might try to learn:
   *   -- how does the dqdx measurement compare when we use the true trajectory?
   *   -- how do the NuSelection variables compare between reco and true trajectories?
   *   -- help set the theoretical maximum performance for the selection.
   */
  class PerfectTruthNuReco : public larcv::larcv_base
  {

  public:
    PerfectTruthNuReco();
    virtual ~PerfectTruthNuReco();

    NuVertexCandidate
      makeNuVertex( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    

    void makeTracks( NuVertexCandidate& nuvtx,
                     const larlite::event_mctrack& ev_mctrack,
                     const larlite::event_larflow3dhit& ev_lm,
                     const std::vector<larcv::Image2D>& adc_v,
                     std::vector<int>& used_v  );

    void makeShowers( NuVertexCandidate& nuvtx,
                      const larlite::event_mcshower& ev_mcshower,
                      const larlite::event_larflow3dhit& ev_lm,
                      const std::vector<larcv::Image2D>& adc_v,
                      const std::vector<larcv::Image2D>& instance_v,                      
                      std::vector<int>& used_v  );

    larutil::SpaceChargeMicroBooNE* _psce;
    
  };
  
  
}
}
    
#endif
