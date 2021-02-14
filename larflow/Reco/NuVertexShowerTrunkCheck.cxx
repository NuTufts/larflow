#include "NuVertexShowerTrunkCheck.h"

namespace larflow {
namespace reco {

  void NuVertexShowerTrunkCheck::checkNuCandidateProngs( larflow::reco::NuVertexCandidate& nuvtx )
  {
  }

  bool NuVertexShowerTrunkCheck::isTrackTrunkOfShower( larlite::track& track,
                                                       larlite::larflowcluster& track_hitcluster,
                                                       larlite::track&          shower_trunk,
                                                       larlite::larflowcluster& shower_hitcluster,
                                                       larlite::pcaxis& shower_pcaxis )
  {
    return false;
  }

  larlite::larflowcluster
  NuVertexShowerTrunkCheck::makeMissingTrunkHits( const std::vector<larcv::Image2D>& adc_v,
                                                  larlite::track& shower_trunk,                                                  
                                                  larlite::larflowcluster& shower_hitcluster,
                                                  larlite::pcaxis& shower_pcaxis )
  {
    larlite::larflowcluster added_hit_v;
    
    return added_hit_v;
  }
  
  
}
}
