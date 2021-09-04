#ifndef __LARFLOW_RECO_NUSHOWER_BUILDER_H__
#define __LARFLOW_RECO_NUSHOWER_BUILDER_H__

#include <vector>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/DataFormat/storage_manager.h"
#include "NuVertexCandidate.h"


namespace larflow {
namespace reco {

  /** 
   * @ingroup NuShowerBuilder
   * @class NuShowerBuilder
   * @brief Build tracks by assembling clusters, starting from neutrino vertices
   *
   * Inherits from TrackClusterBuilder. The base class provides the track buiding algorithms.
   * This class provides interface to the NuVertexCandidate inputs.
   *
   */
  class NuShowerBuilder : public larcv::larcv_base {

  public:

    NuShowerBuilder()
      : larcv::larcv_base("NuShowerBuilder")
    {};
    virtual ~NuShowerBuilder() {};


    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  std::vector<NuVertexCandidate>& nu_candidate_v );

  };

}
}


#endif
