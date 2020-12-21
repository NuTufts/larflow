#ifndef __LARFLOW_RECO_SHOWER_TRUTH_RECO_ANA_H__
#define __LARFLOW_RECO_SHOWER_TRUTH_RECO_ANA_H__

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "larflow/Reco/NuVertexCandidate.h"

namespace larflow {
namespace reco {

  
  class ShowerTruthRecoInfo {
    
  public:
    
    ShowerTruthRecoInfo() {};
    virtual ~ShowerTruthRecoInfo() {};

    int matched_true_trackid;
    int matched_true_pid;
    float matched_nlogl;
    float trueshower_completeness;
    float dist_to_showerstart;
    float cos_true_reco_dir;
    
  };

  class VertexShowerTruthRecoInfo {
    
  public:
    
    VertexShowerTruthRecoInfo() {};
    virtual ~VertexShowerTruthRecoInfo() {};

    int vtxid;
    std::vector< ShowerTruthRecoInfo > showerinfo_v;
    
  };
  
  /**
   * @ingroup Reco
   * @class ShowerTruthRecoAna
   * @brief Class to match truth to reco showers for performance analysis
   *
   * We study the showers stored in the NuVertexCandidate class.
   *
   */  
  class ShowerTruthRecoAna : public larcv::larcv_base {

  public:
    
    ShowerTruthRecoAna()
      : larcv::larcv_base("ShowerTruthRecoAna")
      {};
    virtual ~ShowerTruthRecoAna() {};

    
    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  larflow::reco::NuVertexCandidate& nuvertex_v );
    
  protected:

    
    
  };
  
}
}

#endif
