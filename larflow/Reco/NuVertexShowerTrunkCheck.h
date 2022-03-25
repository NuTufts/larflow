#ifndef __LARFLOW_RECO_NUVERTEX_SHOWER_TRUNK_CHECK_H__
#define __LARFLOW_RECO_NUVERTEX_SHOWER_TRUNK_CHECK_H__

#include <vector>

#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/pcaxis.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"

#include "NuVertexCandidate.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  /**
   * @brief Merge track into shower if consistent with shower trunk
   *
   * This class provides two algorithms related to shower trunk repair.
   *
   * Fixes error due to shower trunks sometimes being labeled track-like.
   * Does this by merging track into the shower object.
   *
   * Checks for and repairs missing shower trunk from vertex to shower start.
   * This can occur when trunk goes through large dead regions.
   * We make 3D points along the vertex to shower start line. The charge
   * will be assigned using the non-dead plane.
   *
   * Ultimately, want to provide a good shower trunk for good 
   * gamma tagging of showers. This menas providing good dQ/dx for electron/gamma
   * PID and for correct gap distances.
   * 
   */
  class NuVertexShowerTrunkCheck : public larcv::larcv_base {

  public:

    NuVertexShowerTrunkCheck()
      : larcv::larcv_base("NuVertexShowerTrunkCheck")
      {};

    virtual ~NuVertexShowerTrunkCheck() {};

    void checkNuCandidateProngs( larflow::reco::NuVertexCandidate& nuvtx );

    void checkNuCandidateProngsForMissingCharge( larflow::reco::NuVertexCandidate& nuvtx,
                                                 larcv::IOManager& iolcv,
                                                 larlite::storage_manager& ioll );
    

    bool isTrackTrunkOfShower( const std::vector<float>& vtxpos,
                               larlite::track& track,
                               larlite::larflowcluster& track_hitcluster,
                               larlite::track&          shower_trunk,
                               larlite::larflowcluster& shower_hitcluster,
                               larlite::pcaxis& shower_pcaxis,
                               float& frac_path, float& frac_core );

    larlite::larflowcluster makeMissingTrunkHits( larflow::reco::NuVertexCandidate& nuvtx, 
                                                  const std::vector<larcv::Image2D>& adc_v,
                                                  larlite::track& shower_trunk,                                                  
                                                  larlite::larflowcluster& shower_hitcluster,
                                                  larlite::pcaxis& shower_pcaxis,
                                                  float& max_gapdist );

  protected:

    void _addTrackAsNewTrunk(  const std::vector<float>& vtxpos,
                               larlite::track& track,
                               larlite::larflowcluster& track_hitcluster,
                               larlite::track&          shower_trunk,
                               larlite::larflowcluster& shower_hitcluster,
                               larlite::pcaxis& shower_pcaxis );
    
    void _addTrackToCore(  const std::vector<float>& vtxpos,
                           larlite::track& track,
                           larlite::larflowcluster& track_hitcluster,
                           larlite::track&          shower_trunk,
                           larlite::larflowcluster& shower_hitcluster,
                           larlite::pcaxis& shower_pcaxis );

    void _mergeNewTrunkHits( const std::vector<float>& vtxpos,
                             const larlite::larflowcluster& newhits,
                             larlite::track& shower_trunk,                                                  
                             larlite::larflowcluster& shower_hitcluster,
                             larlite::pcaxis& shower_pcaxis );
    
  };
  
}
}
    
#endif
