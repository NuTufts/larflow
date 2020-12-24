#include "ShowerTruthRecoAna.h"

namespace larflow {
namespace reco {

  /**
   * @brief process event data
   *
   * @param[in] iolcv Event data contained in LArCV iomanager
   * @param[in] ioll  Event data contained in larlite storage_manager
   * @param[in] nuvertex_v Neutrino interactions reconstructed from the event data above
   */
  void ShowerTruthRecoAna::process( larcv::IOManager& iolcv,
                                    larlite::storage_manager& ioll,
                                    larflow::reco::NuVertexCandidate& nuvertex_v )

  {

    // first we need to extract a list of the true showers

    // then we loop over reco files and assign a truth shower to it

    // then the truth reco metrics are calculated
    // for each reco:
    //   - distance between start point
    //   - cos(0) between truth and reco direction
    //   - store true energy and reco charge sum
    
  }
  
}
}
