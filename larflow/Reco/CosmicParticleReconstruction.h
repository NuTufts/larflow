#ifndef __LARFLOW_RECO_COSMIC_PARTICLE_RECONSTRUCTION_H__
#define __LARFLOW_RECO_COSMIC_PARTICLE_RECONSTRUCTION_H__

#include <string>
#include <vector>

#include "larlite/DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "larflow/Reco/CosmicParticleCandidate.h"
#include "larflow/Reco/KPCluster.h"

namespace larflow {
namespace reco {

  class CosmicParticleReconstruction : public larcv::larcv_base {
  public:

    CosmicParticleReconstruction()
      : larcv::larcv_base("CosmicParticleReconstruction") {};
    ~CosmicParticleReconstruction() {};
    
    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll );   

    void clear();

    void set_default_param_values();

  protected:

    void prepSpacepoints( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void recoKeypoints( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    std::string _flash_producer;
    std::string _wireimg_producer;
    std::string _outoftime_tagged_pixels_producer;
    std::string _larmatch_hit_producer;

    std::vector< larflow::reco::CosmicParticleCandidate > _cosmic_candidates_v;

    // storage for keypoint clusters
    std::vector< larflow::reco::KPCluster > _event_kpc_track_start_v;
    std::vector< larflow::reco::KPCluster > _event_kpc_track_end_v;

  };
  
}
}

#endif
