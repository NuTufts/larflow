#ifndef __LARFLOW_RECO_COSMIC_PARTICLE_CANDIDATE_H__
#define __LARFLOW_RECO_COSMIC_PARTICLE_CANDIDATE_H__

#include <vector>

#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/track.h"
#include "larflow/Reco/KPCluster.h"
#include "ublarcvapp/ParticleToPixelUtils/TrackToSpacePoints.h"

namespace larflow {
namespace reco {

class CosmicParticleCandidate {

public:

  CosmicParticleCandidate()
  {};

  ~CosmicParticleCandidate()
  {};

  larflow::reco::KPCluster startpt;        ///< Reconstructed Keypoint used to seed cosmic particle
  larlite::larflowcluster particle_hits_v; ///< larmatch hits associated to this particle candidate
  larlite::track particle_track;           ///< line-segment representation of the particle trajectory
  std::vector< ublarcvapp::pixelutils::TrackToSpacePoints::SpacePointCharge > ptpixcharge_v; ///< association between spacepoint and pixel info

  float totalpe_prediction; ///< predicted total pe
  float totalpe_observed;   ///< total pe for the matched observed flash
  std::vector< float > flashpdf_prediction; ///< predicted flash pdf (i.e. normalized pe)
  std::vector< float > flashpdf_observed_matched; ///< matched observe flash pde (i.e. normalized pe)
  float t0_matched;  ///< true t0 of the interaction, relative to beam trigger
  float dt0_matched; ///< time difference between when flash occurred and the start of the track reaches the wireplanes


};

}
}

#endif