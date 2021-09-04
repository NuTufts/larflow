#ifndef __LARFLOW_CRTMATCH_CRTMATCH_H__
#define __LARFLOW_CRTMATCH_CRTMATCH_H__

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "CRTTrackMatch.h"
#include "CRTHitMatch.h"

namespace larflow {
namespace crtmatch {

  /**
   * @ingroup CRTMatch
   * @class CRTMatch
   * @brief Perform track matching to CRT tracks and CRT hits
   *
   * Class that runs both CRTTrackMatch and CRTHitMatch.
   *
   */
  class CRTMatch {

  public:
    
    CRTMatch() {};
    virtual ~CRTMatch() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll);
    void store_output( larcv::IOManager& outlcv, larlite::storage_manager& outll, bool remove_if_no_flash=true );
    
    CRTTrackMatch _track_matcher; ///< Algo for CRT-track to wire image ionization matching
    CRTHitMatch   _hit_matcher;   ///< Algo for CRT-hit to spacepoint cluster matching

    std::vector< larcv::Image2D > untagged_v;    ///< image where matched pixels are removed
    std::vector< larcv::Image2D > track_index_v; ///< image where crt track index labels image, so we can match larflow clusters to it
    std::vector< larlite::larflowcluster > _unmatched_clusters_v; ///< clusters not matched to crthit or crttracks
    
  };
  
}
}

#endif
