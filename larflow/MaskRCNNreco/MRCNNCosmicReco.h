#ifndef __LARFLOW_MASKRCNNReco_MRCNNCosmicReco_h__
#define __LARFLOW_MASKRCNNReco_MRCNNCosmicReco_h__

#include "larlite/DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "larflow/Reco/KPSRecoManager.h"

namespace larflow {
namespace mrcnnreco {

  /**
   * @brief reco workflow to reconstruct cosmic muons for sideband and calibration purposes
   *
   *
   */
  class MRCNNCosmicReco : public larflow::reco::KPSRecoManager  {

  public:
    
    MRCNNCosmicReco( std::string output_file )
      : larflow::reco::KPSRecoManager(output_file,2,"MRCNNCosmicReco")
    {};
    virtual ~MRCNNCosmicReco() {};
    
    void process( larcv::IOManager& lcvio, larlite::storage_manager& ioll );
    
  };
  
  
}
}

#endif
