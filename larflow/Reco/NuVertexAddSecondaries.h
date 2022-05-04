#ifndef __NuVertexAddSecondaries_H__
#define __NuVertexAddSecondaries_H__

#include "larlite/DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"
#include "NuVertexCandidate.h"
#include "ClusterBookKeeper.h"

namespace larflow {
namespace reco {

  /**
   * @brief Attach remaining clusters to nuvertex prongs
   *
   */ 
  class NuVertexAddSecondaries : public larcv::larcv_base {
  public:

    NuVertexAddSecondaries()
      : larcv::larcv_base("NuVertexAddSecondaries")
      {};

    virtual ~NuVertexAddSecondaries() {};

    void process( larflow::reco::NuVertexCandidate& nuvtx,
		  larflow::reco::ClusterBookKeeper& nuclusterbook,		  
		  larlite::storage_manager& ioll );
    

    float testTrackTrackIntersection( larlite::track& track,
				      larlite::pcaxis& cluster_pca,
				      const float _min_line_dist );
    
  };
  
}
}

#endif
